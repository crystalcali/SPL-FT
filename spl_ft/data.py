"""Streaming, bounded disk sampling, grouped splits, native LFM3 loss masks."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from .config import Experiment, MODEL, home


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalize(row):
    raw = row.get("messages", row.get("conversations"))
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise ValueError("Expected conversations/messages list")
    aliases = {"human": "user", "gpt": "assistant", "observation": "tool"}
    messages = []
    for m in raw:
        role = aliases.get(m.get("role", m.get("from")), m.get("role", m.get("from")))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role {role}")
        item = {"role": role, "content": m.get("content", m.get("value", "")) or ""}
        for key in ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking", "reasoning"):
            if m.get(key) is not None:
                item[key] = m[key]
        for call in item.get("tool_calls", []):
            fn = call.get("function", call)
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
        messages.append(item)
    users = [m for m in messages if m["role"] == "user"]
    if not users or not any(m["role"] == "assistant" for m in messages):
        raise ValueError("Missing task or assistant response")
    task = users[0]["content"]
    if isinstance(task, str) and "Task Description:" in task:
        task = task.split("Task Description:", 1)[1]
        task = task.split("Current terminal state:", 1)[0]
    text = task if isinstance(task, str) else json.dumps(task, ensure_ascii=False)
    prompt = re.sub(r"\s+", " ", text).strip().casefold()
    # Task names catch corpus duplicates; prompt grouping also catches distinct source IDs.
    group = digest(prompt)
    return {"id": digest(messages), "group": group, "messages": messages,
            "tools": row.get("tools") or [], "source": str(row.get("source", "")),
            "agent": str(row.get("agent", ""))}


class Pools:
    """Keep bottom-k distinct prompt groups in two independently seeded pools.

    holdout is independent of train_size. train has enough reserves to remove all
    holdouts after scanning. One deterministic variant per prompt limits leakage.
    """
    def __init__(self, path, config):
        self.c = config
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS pool(kind TEXT,g TEXT,rank TEXT,rid TEXT,payload TEXT,PRIMARY KEY(kind,g))")
        self.db.execute("CREATE INDEX IF NOT EXISTS ranks ON pool(kind,rank)")
        self.limits = {"holdout": config.test_size + config.val_size,
                       "train": config.train_size + config.test_size + config.val_size}
        self.cutoffs = {k: None for k in self.limits}
        self.counts = {k: 0 for k in self.limits}

    def add(self, row):
        payload = None
        for kind, limit in self.limits.items():
            seed = self.c.split_seed if kind == "holdout" else self.c.sample_seed
            rank = digest([seed, row["group"]])
            cutoff = self.cutoffs[kind]
            if cutoff is not None and rank > cutoff:
                continue
            existing = self.db.execute("SELECT rid FROM pool WHERE kind=? AND g=?", (kind, row["group"])).fetchone()
            if existing and existing[0] <= row["id"]:
                continue
            payload = payload or json.dumps(row, ensure_ascii=False)
            self.db.execute("INSERT OR REPLACE INTO pool VALUES(?,?,?,?,?)", (kind, row["group"], rank, row["id"], payload))
            if not existing:
                self.counts[kind] += 1
            if self.counts[kind] > limit:
                self.db.execute("DELETE FROM pool WHERE kind=? AND g=(SELECT g FROM pool WHERE kind=? ORDER BY rank DESC LIMIT 1)", (kind, kind))
                self.counts[kind] -= 1
            if self.counts[kind] == limit:
                self.cutoffs[kind] = self.db.execute("SELECT rank FROM pool WHERE kind=? ORDER BY rank DESC LIMIT 1", (kind,)).fetchone()[0]

    def finish(self):
        self.db.commit()
        selected = {}
        holds = [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM pool WHERE kind='holdout' ORDER BY rank")]
        selected["test"] = holds[:self.c.test_size]
        selected["validation"] = holds[self.c.test_size:]
        reserved = {r["group"] for r in holds}
        selected["train"] = [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM pool WHERE kind='train' ORDER BY rank") if json.loads(r[0])["group"] not in reserved][:self.c.train_size]
        expected = {"test": self.c.test_size, "validation": self.c.val_size, "train": self.c.train_size}
        self.db.close()
        for k, n in expected.items():
            if len(selected[k]) < n:
                raise ValueError(f"Only {len(selected[k])} distinct {k} tasks available; requested {n}. Increase scan limit or reduce sample sizes.")
        return selected


def prepare(config: Experiment, root: Path, emit=print):
    from huggingface_hub import HfApi
    from datasets import load_dataset
    revision = config.dataset_revision or HfApi().dataset_info(config.dataset).sha
    identity = {"dataset": config.dataset, "revision": revision, "configs": sorted(config.td1_configs) if config.preset != "reasoning" else ["default"],
                "train_size": config.train_size, "val_size": config.val_size, "test_size": config.test_size,
                "split_seed": config.split_seed, "sample_seed": config.sample_seed, "scan_limit": config.scan_limit, "schema": 2}
    folder = root / "datasets" / digest(identity)[:20]
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    dbpath = folder / "sampling.sqlite3"
    dbpath.unlink(missing_ok=True)
    pools = Pools(dbpath, config)
    scanned = invalid = 0
    for subset in identity["configs"]:
        ds = load_dataset(config.dataset, subset, split="train", revision=revision, streaming=True)
        for raw in ds:
            if config.scan_limit is not None and scanned >= config.scan_limit:
                break
            scanned += 1
            try:
                row = normalize(raw)
                pools.add(row)
            except (ValueError, TypeError, KeyError):
                invalid += 1
            if scanned % 1000 == 0:
                pools.db.commit()
                emit(f"Scanned {scanned:,}; rejected {invalid:,}; subset {subset}")
        if config.scan_limit is not None and scanned >= config.scan_limit:
            break
    selected = pools.finish()
    for split, rows in selected.items():
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {**identity, "scanned": scanned, "invalid": invalid, "prefix_sample": config.scan_limit is not None,
                "split_ids": {k: [r["id"] for r in v] for k, v in selected.items()},
                "split_groups": {k: [r["group"] for r in v] for k, v in selected.items()}}
    temp = folder / "manifest.tmp"
    temp.write_text(json.dumps(manifest, indent=2))
    temp.replace(manifest_path)
    dbpath.unlink(missing_ok=True)
    emit(f"Dataset ready: {folder}; scanned {scanned:,} records")
    return folder


def tokenize_row(tokenizer, row):
    enc = tokenizer.apply_chat_template(row["messages"], tools=row.get("tools") or None,
        tokenize=True, add_generation_prompt=False, return_dict=True,
        return_assistant_tokens_mask=True, preserve_thinking=True)
    mask = enc.get("assistant_masks", enc.get("assistant_tokens_mask"))
    if mask is None or not any(mask):
        raise ValueError("Chat template did not return a usable assistant generation mask")
    ids = list(enc["input_ids"])
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [t if m else -100 for t, m in zip(ids, mask)]}


def windows(tokenizer, row, max_length):
    """Full trajectory if it fits; otherwise per-decision history windows.

    Keep the task plus whole recent messages. Each target assistant turn is
    supervised once. Oversize individual answers are skipped, never cut in half.
    """
    full = tokenize_row(tokenizer, row)
    if len(full["input_ids"]) <= max_length:
        return [full], 0
    messages = row["messages"]
    first_user = next(i for i, m in enumerate(messages) if m["role"] == "user")
    base = messages[:first_user + 1]
    result, dropped = [], 0
    start = first_user + 1
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        while True:
            context = base + messages[start:i + 1]
            enc = tokenize_row(tokenizer, {**row, "messages": context})
            if len(enc["input_ids"]) <= max_length:
                break
            if start >= i:
                enc = None
                dropped += 1
                break
            start += 1
            # Avoid starting the retained suffix with an orphaned tool result.
            while start < i and messages[start]["role"] == "tool":
                start += 1
        if enc:
            # Use the last native generation span. Tokenizing a prefix separately
            # can merge the trailing newline differently at the next special token.
            supervised = [n for n, value in enumerate(enc["labels"]) if value != -100]
            last = supervised[-1]
            begin = last
            while begin > 0 and enc["labels"][begin - 1] != -100:
                begin -= 1
            enc["labels"][:begin] = [-100] * begin
            if any(t != -100 for t in enc["labels"]):
                result.append(enc)
    return result, dropped


def tokenize_splits(folder, tokenizer, config, emit=print):
    from datasets import Dataset
    token_dir = folder / ("tokens-" + digest([config.model_revision, config.max_length, 2])[:16])
    token_dir.mkdir(exist_ok=True)
    stats_path = token_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        return {s: Dataset.from_json(str(token_dir / f"{s}.jsonl")) for s in stats}, stats
    splits, stats = {}, {}
    for split in ("train", "validation", "test"):
        dropped = count = tokens = targets = maximum = 0
        lengths = []
        output = token_dir / f"{split}.jsonl"
        with (folder / f"{split}.jsonl").open() as source, output.open("w") as destination:
            for line in source:
                row = json.loads(line)
                ws, n = windows(tokenizer, row, config.max_length)
                dropped += n
                for window in ws:
                    destination.write(json.dumps(window) + "\n")
                    count += 1
                    width = len(window["input_ids"])
                    tokens += width
                    targets += sum(t != -100 for t in window["labels"])
                    maximum = max(maximum, width)
                    if len(lengths) < 10000:
                        lengths.append(width)
        if not count:
            raise ValueError(f"No usable {split} examples fit max_length={config.max_length}")
        splits[split] = Dataset.from_json(str(output))
        stats[split] = {"windows": count, "dropped_assistant_turns": dropped,
                        "tokens": tokens, "target_tokens": targets, "lengths_preview": lengths,
                        "max_length": maximum, "mean_length": tokens / count}
        emit(f"{split}: {count} windows, {tokens:,} tokens, {targets:,} targets; dropped {dropped}")
    stats_path.write_text(json.dumps(stats, indent=2))
    return splits, stats
