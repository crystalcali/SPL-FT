from pathlib import Path
from typing import Literal
import os
import platform
from pydantic import BaseModel, Field, ConfigDict, model_validator

MODEL = "LiquidAI/LFM3-1.7B"
# Pin the remote modeling code as well as weights/tokenizer.
MODEL_REVISION = "1da06bec6a819da10b54c67e5c90972b46d81871"
TD1 = "nvidia/Nemotron-Terminal-Corpus"
TD2 = "open-thoughts/OpenThoughts3-1.2M"
BENCH_REPO = "https://huggingface.co/datasets/harborframework/terminal-bench-2.1"


def home() -> Path:
    p = Path(os.environ.get("SPL_FT_HOME", ".spl-ft")).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Terminal SFT", min_length=1, max_length=120)
    preset: Literal["terminal", "long_horizon", "reasoning"] = "terminal"
    method: Literal["lora", "full"] = "lora"
    train_size: int = Field(default=2000, ge=1, le=1000000)
    val_size: int = Field(default=100, ge=1, le=10000)
    test_size: int = Field(default=100, ge=1, le=10000)
    split_seed: int = 42
    sample_seed: int = 123
    training_seed: int = 42
    max_length: int = Field(default=4096, ge=512, le=131072)
    epochs: float = Field(default=1, gt=0, le=50)
    max_steps: int = Field(default=-1, ge=-1, le=1000000)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.01)
    micro_batch: int = Field(default=1, ge=1, le=32)
    accumulation: int = Field(default=16, ge=1, le=1024)
    lora_rank: int = Field(default=16, ge=1, le=256)
    save_steps: int = Field(default=100, ge=1)
    model_revision: str = MODEL_REVISION
    dataset_revision: str | None = None
    td1_configs: list[Literal["dataset_adapters", "skill_based_easy", "skill_based_medium", "skill_based_mixed"]] = Field(default_factory=lambda: ["dataset_adapters", "skill_based_easy", "skill_based_medium", "skill_based_mixed"])
    # None scans the entire pinned corpus: no biased prefix presented as a random sample.
    scan_limit: int | None = Field(default=None, ge=100)
    resume_from: str | None = None
    auto_benchmark: bool = False

    @model_validator(mode="after")
    def validate_config(self):
        if self.max_steps == 0:
            raise ValueError("max_steps must be -1 or positive")
        if self.preset != "reasoning" and not self.td1_configs:
            raise ValueError("Select at least one TD1 subset")
        return self

    @property
    def dataset(self):
        return TD2 if self.preset == "reasoning" else TD1


def presets():
    return {
        "terminal": Experiment().model_dump(),
        "long_horizon": Experiment(name="Long-horizon SFT", preset="long_horizon", max_length=16384, learning_rate=0.00005).model_dump(),
        "reasoning": Experiment(name="Reasoning SFT", preset="reasoning", max_length=8192).model_dump(),
    }


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Original · Terminal-Bench 2.1", max_length=120)
    model_run: str | None = None
    agent: Literal["opencode", "terminus-2"] = "terminus-2"
    skill_ids: list[str] = Field(default_factory=list)
    task_names: list[str] = Field(default_factory=list)
    n_tasks: int | None = Field(default=None, ge=1, le=89)
    attempts: int = Field(default=1, ge=1, le=10)
    concurrency: int = Field(default=1, ge=1, le=8)
    max_turns: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_new_tokens: int = Field(default=4096, ge=256, le=32768)
    context_length: int = Field(default=32768, ge=4096, le=262144)
    seed: int = 42
    benchmark_revision: str | None = None
    architecture: Literal["native", "amd64"] = Field(default_factory=lambda: "amd64" if platform.machine() in ("aarch64", "arm64") else "native")
    keep_environments: bool = False

    @model_validator(mode="after")
    def room_for_generation(self):
        if self.max_new_tokens >= self.context_length:
            raise ValueError("Generation budget must be smaller than context length")
        return self
