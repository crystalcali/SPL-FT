import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
import httpx
from .config import Benchmark, MODEL, MODEL_REVISION, BENCH_REPO
from .data import digest
from .store import Store


def summarize(results, expected):
    trials = []
    for r in results:
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        if reward is None and len(rewards) == 1:
            reward = next(iter(rewards.values()))
        exception = r.get("exception_info")
        valid = r.get("reward_valid", not exception) and isinstance(reward, (int, float))
        context = r.get("agent_result") or {}
        trials.append({"task": r.get("task_name", "unknown"), "trial": r.get("trial_name"),
                       "reward": reward if valid else None, "passed": bool(valid and reward == 1),
                       "error": (exception or {}).get("exception_type"), "reward_valid": bool(valid),
                       "input_tokens": context.get("n_input_tokens"), "output_tokens": context.get("n_output_tokens")})
    valid = [t for t in trials if t["reward_valid"]]
    passed = sum(t["passed"] for t in trials)
    return {"expected": expected, "finished": len(trials), "graded": len(valid), "passed": passed,
            "errors": sum(not t["reward_valid"] for t in trials), "complete": len(trials) == expected and len(valid) == expected,
            "success_rate": passed / expected if expected else None,
            "graded_success_rate": passed / len(valid) if valid else None,
            "trials": trials,
            "input_tokens": sum(t["input_tokens"] or 0 for t in trials),
            "output_tokens": sum(t["output_tokens"] or 0 for t in trials)}


def skill_snapshot(root, skill_ids, output):
    import shutil
    manifest, paths = [], []
    for sid in skill_ids:
        if not sid.isalnum():
            raise ValueError("Invalid skill identifier")
        source = root / "skills" / sid
        meta = json.loads((source / "metadata.json").read_text())
        destination = output / "skills" / sid
        shutil.copytree(source / "content", destination)
        paths.append(str(destination.resolve()))
        manifest.append({"id": sid, **meta})
    return paths, manifest


def harbor_config(c, folder, tasks, endpoint, skill_paths, agent_version=None):
    agent = {"name": c.agent, "model_name": "openai/lfm3", "base_url": endpoint,
             "model_api_key_name": "SPL_INFER_KEY", "model_api_format": "openai_chat_completions", "skills": skill_paths}
    if c.agent == "opencode":
        agent["kwargs"] = {"opencode_config": {"provider": {"openai": {"models": {"lfm3": {
            "name": "LFM3", "limit": {"context": c.context_length, "output": c.max_new_tokens},
            "options": {"temperature": c.temperature}}}}}}}
    else:
        agent["kwargs"] = {"max_turns": c.max_turns, "temperature": c.temperature,
            "model_info": {"max_input_tokens": c.context_length - c.max_new_tokens, "max_output_tokens": c.max_new_tokens},
            "llm_call_kwargs": {"max_tokens": c.max_new_tokens}, "interleaved_thinking": True}
    if c.agent == "opencode" and agent_version:
        agent["kwargs"]["version"] = agent_version
    overlay = folder / "network.json"
    service = {"extra_hosts": ["host.docker.internal:host-gateway"]}
    if c.architecture == "amd64":
        service["platform"] = "linux/amd64"
    overlay.write_text(json.dumps({"services": {"main": service}}))
    return {"job_name": "evaluation", "jobs_dir": str(folder / "harbor"), "n_attempts": c.attempts,
            "n_concurrent_trials": c.concurrency, "agents": [agent],
            "tasks": [t.model_dump(mode="json") if hasattr(t, "model_dump") else t for t in tasks],
            "environment": {"type": "docker", "delete": not c.keep_environments,
                            "extra_docker_compose": [str(overlay.resolve())]}, "retry": {"max_retries": 0}}


def run_benchmark(c: Benchmark, folder: Path):
    from huggingface_hub import HfApi, snapshot_download
    from harbor.models.job.config import DatasetConfig, JobConfig
    from harbor.job import Job
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "compose", "version"], check=True)
    root = Store().root
    from .tasks import environment
    (folder / "environment.json").write_text(json.dumps(environment(), indent=2))
    revision = c.benchmark_revision or HfApi().dataset_info("harborframework/terminal-bench-2.1").sha
    # Materialize real files: HF cache snapshot symlinks intentionally fail
    # Harbor's task-root containment validation.
    bench_path = snapshot_download("harborframework/terminal-bench-2.1", repo_type="dataset", revision=revision,
                                   local_dir=root / "benchmarks" / revision)
    task_root = Path(bench_path) / "tasks"
    available = asyncio.run(DatasetConfig(path=task_root).get_task_configs())
    declared = [p for p in task_root.iterdir() if p.is_dir() and (p / "task.toml").exists()]
    if len(available) != len(declared):
        raise RuntimeError(f"Harbor accepted {len(available)}/{len(declared)} task directories; refusing to silently omit invalid environments")
    dataset = DatasetConfig(path=task_root, task_names=[n.removeprefix("terminal-bench/") for n in c.task_names] or None, n_tasks=c.n_tasks)
    tasks = asyncio.run(dataset.get_task_configs())
    if not tasks:
        raise ValueError("No benchmark tasks selected")
    paths, skills = skill_snapshot(root, c.skill_ids, folder)
    model_path, model_revision = MODEL, MODEL_REVISION
    if c.model_run:
        parent = Store().get(c.model_run)
        if parent["kind"] != "train" or parent["state"] != "succeeded":
            raise ValueError("Select a successfully trained model")
        artifact = root / "runs" / c.model_run / "model"
        if not (artifact / "config.json").exists() and not (artifact / "adapter_config.json").exists():
            raise ValueError("Training artifact missing")
        model_path = str(artifact)
        model_revision = parent["config"]["model_revision"]
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    server_config = {"model_path": model_path, "revision": model_revision, "port": port,
                     "context_length": c.context_length, "temperature": c.temperature,
                     "max_new_tokens": c.max_new_tokens, "seed": c.seed}
    (folder / "inference.json").write_text(json.dumps(server_config, indent=2))
    env = {**os.environ, "SPL_INFER_KEY": key}
    if c.architecture == "amd64":
        env["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"
        os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"
    os.environ["SPL_INFER_KEY"] = key
    import importlib.metadata
    import re
    harbor_version = importlib.metadata.version("harbor")
    agent_version = "2.0.0"  # Terminus bundled with the pinned Harbor release.
    if c.agent == "opencode":
        # Resolve once and pin every container in this run to the same CLI.
        response = httpx.get("https://registry.npmjs.org/opencode-ai/latest", timeout=30)
        response.raise_for_status()
        agent_version = response.json()["version"]
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", agent_version):
            raise ValueError("Unexpected OpenCode package version")
    config = harbor_config(c, folder, tasks, f"http://{'host.docker.internal' if c.agent == 'opencode' else '127.0.0.1'}:{port}/v1", paths, agent_version)
    # Terminus runs on the host; OpenCode runs inside the task container.
    # The compose overlay supplies host.docker.internal on native Linux.
    (folder / "harbor-config.json").write_text(json.dumps(config, indent=2))
    provenance = {"benchmark_repo": BENCH_REPO, "benchmark_revision": revision, "model_revision": model_revision,
                  "skills": skills, "agent_version": agent_version, "harbor_version": harbor_version,
                  "tasks": config["tasks"], "expected": len(tasks) * c.attempts,
                  "comparison_key": digest({"revision": revision, "tasks": config["tasks"], "skills": skills,
                     "agent": c.agent, "agent_version": agent_version, "harbor_version": harbor_version, "temperature": c.temperature, "context": c.context_length,
                     "output": c.max_new_tokens, "architecture": c.architecture, "attempts": c.attempts,
                     "max_turns": c.max_turns, "seed": c.seed})}
    (folder / "benchmark-manifest.json").write_text(json.dumps(provenance, indent=2))
    print(f"Loading {model_path}; {len(tasks)} tasks × {c.attempts} attempts", flush=True)
    start = time.monotonic()
    with open(folder / "inference.log", "w") as log:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "spl_ft.inference", str(folder / "inference.json")], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(timeout=3, trust_env=False, headers={"Authorization": f"Bearer {key}"}) as client:
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError("Inference process failed; inspect inference.log")
                    try:
                        if client.get(f"http://127.0.0.1:{port}/v1/models").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("Model server did not become ready in 15 minutes")
            async def execute():
                job = await Job.create(JobConfig.model_validate(config))
                await job.run()
            asyncio.run(execute())
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            records = []
            for p in (folder / "harbor").rglob("result.json"):
                try:
                    r = json.loads(p.read_text())
                    if "task_name" in r and "trial_name" in r:
                        r["_path"] = str(p.relative_to(folder))
                        records.append(r)
                except (ValueError, OSError):
                    continue
            summary = summarize(records, provenance["expected"])
            summary.update(seconds=time.monotonic() - start, **provenance)
            (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise RuntimeError("Benchmark incomplete or has ungraded trials; see summary and Harbor logs")
    print(f"Terminal-Bench: {summary['passed']}/{summary['expected']} = {summary['success_rate']:.2%}")
