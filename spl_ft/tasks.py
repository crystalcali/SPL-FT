import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from .config import Experiment, Benchmark, MODEL, home
from .data import prepare, tokenize_splits


def environment():
    versions = {}
    for package in ("spl-ft", "torch", "transformers", "trl", "peft", "accelerate", "datasets", "harbor"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "architecture": platform.machine(), "versions": versions}


def doctor():
    import importlib.util
    import shutil
    import psutil
    checks = {"environment": environment(), "memory_available_gib": round(psutil.virtual_memory().available / 2**30, 1),
              "disk_free_gib": round(shutil.disk_usage(home()).free / 2**30, 1)}
    for name, command in {"docker": ["docker", "info", "--format", "{{.Architecture}}"],
                          "compose": ["docker", "compose", "version"], "gpu": ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"]}.items():
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=15)
            checks[name] = {"ok": p.returncode == 0, "detail": (p.stdout or p.stderr)[-1600:]}
        except (OSError, subprocess.TimeoutExpired) as e:
            checks[name] = {"ok": False, "detail": str(e)}
    if importlib.util.find_spec("torch"):
        try:
            import torch
            checks["cuda"] = {"ok": torch.cuda.is_available(), "detail": str(torch.version.cuda)}
        except Exception as e:
            checks["cuda"] = {"ok": False, "detail": str(e)}
    else:
        checks["cuda"] = {"ok": False, "detail": "PyTorch not installed; run bash scripts/setup.sh"}
    if platform.machine() in ("aarch64", "arm64"):
        checks["amd64_emulation"] = {"ok": Path("/proc/sys/fs/binfmt_misc/qemu-x86_64").exists(),
                                     "detail": "For original x86 task images: bash scripts/enable-amd64.sh"}
    return checks


def train(c, folder):
    import torch
    import psutil
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run Diagnostics and install CUDA PyTorch via scripts/setup.sh")
    if psutil.virtual_memory().available < (30 if c.method == "full" else 12) * 2**30:
        raise RuntimeError("Not enough available memory. Stop other large model servers before training.")
    set_seed(c.training_seed)
    data_path = prepare(c, home())
    manifest = json.loads((data_path / "manifest.json").read_text())
    (folder / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2))
    (folder / "data-path.txt").write_text(str(data_path))
    (folder / "environment.json").write_text(json.dumps(environment(), indent=2))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=c.model_revision, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data, stats = tokenize_splits(data_path, tokenizer, c)
    (folder / "data-stats.json").write_text(json.dumps(stats, indent=2))
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=c.model_revision, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    if c.method == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=c.lora_rank, lora_alpha=2*c.lora_rank, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"], task_type="CAUSAL_LM"))
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    model.to("cuda")

    def collate(features):
        width = max(len(f["input_ids"]) for f in features)
        return {k: torch.tensor([f[k] + [pad]*(width-len(f[k])) for f in features], dtype=torch.long)
                for k, pad in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100))}

    class Metrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            event = {"step": state.global_step, "time": time.time(), **(logs or {}),
                     "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "ram_available_gib": psutil.virtual_memory().available / 2**30}
            with open(folder / "metrics.jsonl", "a") as f:
                f.write(json.dumps(event, default=str) + "\n")

    arguments = TrainingArguments(output_dir=str(folder / "checkpoints"), num_train_epochs=c.epochs,
        max_steps=c.max_steps, per_device_train_batch_size=c.micro_batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=c.accumulation, learning_rate=c.learning_rate, lr_scheduler_type="cosine",
        warmup_ratio=0.05, weight_decay=0.01, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1,
        eval_strategy="steps", eval_steps=c.save_steps, save_strategy="steps", save_steps=c.save_steps,
        save_total_limit=2, seed=c.training_seed, data_seed=c.training_seed, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, prediction_loss_only=True,
        include_num_input_tokens_seen=True, optim="adamw_torch")
    # Native generation masks were applied above; Trainer preserves our labels.
    # Rebuilding text inside an SFT trainer could mask tool observations incorrectly.
    trainer = Trainer(model=model, args=arguments, train_dataset=data["train"], eval_dataset=data["validation"],
                      data_collator=collate, processing_class=tokenizer, callbacks=[Metrics()])
    start = time.monotonic()
    result = trainer.train(resume_from_checkpoint=c.resume_from or None)
    validation = trainer.evaluate(metric_key_prefix="validation")
    test = trainer.evaluate(data["test"], metric_key_prefix="test")
    artifact = folder / "model"
    trainer.save_model(str(artifact))
    tokenizer.save_pretrained(artifact)
    (folder / "summary.json").write_text(json.dumps({**result.metrics, **validation, **test,
        "seconds": time.monotonic()-start, "method": c.method, "preset": c.preset,
        "model_revision": c.model_revision, "dataset_revision": manifest["revision"],
        "train_target_tokens_per_epoch": stats["train"]["target_tokens"],
        "cuda_peak_gib": torch.cuda.max_memory_allocated()/2**30}, indent=2))
    print(f"Training complete. Model saved to {artifact}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["prepare", "train", "benchmark", "doctor"])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    import signal
    import threading
    def terminate(signum, frame):
        raise KeyboardInterrupt("Task cancelled")
    signal.signal(signal.SIGTERM, terminate)
    parent = os.getppid()
    def parent_watchdog():
        while True:
            time.sleep(1)
            if os.getppid() != parent:
                if os.getpgrp() == os.getpid():
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                return
    threading.Thread(target=parent_watchdog, daemon=True).start()
    config = json.loads((args.folder / "config.json").read_text())
    if args.kind == "train":
        train(Experiment.model_validate(config), args.folder)
    elif args.kind == "prepare":
        p = prepare(Experiment.model_validate(config), home())
        (args.folder / "data-path.txt").write_text(str(p))
        (args.folder / "summary.json").write_text((p / "manifest.json").read_text())
    elif args.kind == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(Benchmark.model_validate(config), args.folder)
    elif args.kind == "doctor":
        result = doctor()
        (args.folder / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
