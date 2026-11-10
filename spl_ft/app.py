import hmac
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .config import Experiment, Benchmark, presets
from .store import Store
from .worker import Worker
from .skills import import_skill


def access_token(root):
    if os.getenv("SPL_FT_TOKEN"):
        return os.environ["SPL_FT_TOKEN"]
    path = root / "access-token"
    if not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(24))
    return path.read_text().strip()


def create_app(root=None, start_worker=True):
    db = Store(root)
    token = access_token(db.root)
    worker = Worker(db)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title="SPL-FT", lifespan=lifespan)
    app.state.store = db

    @app.middleware("http")
    async def auth(request, call_next):
        if request.url.path.startswith("/api/") and request.url.path != "/api/login":
            supplied = request.cookies.get("spl_session", "")
            bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not (hmac.compare_digest(supplied, token) or hmac.compare_digest(bearer, token)):
                return JSONResponse({"detail": "Login required"}, status_code=401)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin writes disabled"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.post("/api/login")
    def login(body: dict):
        if not hmac.compare_digest(str(body.get("token", "")), token):
            raise HTTPException(401, "Invalid access key")
        response = JSONResponse({"ok": True})
        response.set_cookie("spl_session", token, httponly=True, samesite="strict", max_age=7*86400)
        return response

    @app.post("/api/logout")
    def logout():
        response = JSONResponse({"ok": True})
        response.delete_cookie("spl_session")
        return response

    @app.get("/api/state")
    def state():
        import platform
        import psutil
        import shutil
        return {"jobs": db.list(), "paused": db.setting("paused", False), "presets": presets(),
                "system": {"architecture": platform.machine(), "memory_available_gib": round(psutil.virtual_memory().available/2**30, 1),
                           "disk_free_gib": round(shutil.disk_usage(db.root).free/2**30, 1)},
                "benchmark_defaults": Benchmark().model_dump()}

    @app.post("/api/experiments")
    def experiment(c: Experiment):
        job = db.enqueue("train", c.model_dump())
        if c.auto_benchmark:
            b = Benchmark(name=c.name + " · Terminal-Bench", model_run=job["id"])
            db.enqueue("benchmark", b.model_dump(), dependency=job["id"])
        return job

    @app.post("/api/prepare")
    def dataset(c: Experiment):
        return db.enqueue("prepare", c.model_dump())

    @app.post("/api/benchmarks")
    def benchmark(c: Benchmark):
        dep = None
        if c.model_run:
            source = db.get(c.model_run)
            if source["kind"] != "train" or source["state"] in {"failed", "cancelled", "interrupted", "blocked"}:
                raise ValueError("Select a successful or queued training run")
            if source["state"] != "succeeded":
                dep = c.model_run
        for sid in c.skill_ids:
            if not sid.isalnum() or not (db.root / "skills" / sid / "metadata.json").exists():
                raise ValueError("Unknown skill")
        return db.enqueue("benchmark", c.model_dump(), dependency=dep)

    @app.post("/api/doctor")
    def doctor():
        return db.enqueue("doctor", {"name": "Диагностика Spark"}, priority=100)

    @app.post("/api/queue")
    def queue(body: dict):
        db.set_setting("paused", bool(body.get("paused")))
        return {"paused": db.setting("paused")}

    @app.get("/api/jobs/{jid}")
    def detail(jid: str):
        job = db.get(jid)
        folder = db.run_dir(jid)
        for name, filename in {"summary": "summary.json", "data_stats": "data-stats.json"}.items():
            p = folder / filename
            if p.exists():
                try:
                    job[name] = json.loads(p.read_text())
                except ValueError:
                    pass
        metrics = folder / "metrics.jsonl"
        job["metrics"] = []
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    job["metrics"].append(json.loads(line))
                except ValueError:
                    continue
        log = folder / "console.log"
        if log.exists():
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - 60000))
                job["log"] = f.read().decode(errors="replace")
        else:
            job["log"] = "Ожидание запуска…"
        job["files"] = [str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".log", ".txt", ".md", ".yaml"} and "model" not in p.parts and "checkpoints" not in p.parts]
        job["checkpoints"] = [str(p.relative_to(folder)) for p in sorted((folder / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) if (p / "trainer_state.json").exists()]
        return job

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str):
        job = db.get(jid)
        if job["state"] == "queued":
            db.update(jid, state="cancelled", cancel=1, finished=time.time())
        elif job["state"] == "running":
            db.update(jid, cancel=1)
        return db.get(jid)

    @app.post("/api/jobs/{jid}/priority")
    def priority(jid: str, body: dict):
        db.get(jid)
        db.update(jid, priority=max(-100, min(100, int(body.get("priority", 0)))))
        return db.get(jid)

    @app.post("/api/jobs/{jid}/retry")
    def retry(jid: str, body: dict):
        job = db.get(jid)
        if job["state"] in {"running", "queued"}:
            raise ValueError("Stop the current run before retrying")
        c = job["config"].copy()
        c["name"] += " · retry"
        if body.get("resume"):
            if job["kind"] != "train":
                raise ValueError("Checkpoint resume is supported for training")
            checkpoints = sorted((db.run_dir(jid) / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            checkpoints = [p for p in checkpoints if (p / "trainer_state.json").exists()]
            if not checkpoints:
                raise ValueError("No complete checkpoint available")
            c["resume_from"] = str(checkpoints[-1])
        elif job["kind"] == "train":
            c["resume_from"] = None
        return db.enqueue(job["kind"], c)

    @app.get("/api/jobs/{jid}/file/{name:path}")
    def file(jid: str, name: str):
        root = db.run_dir(jid).resolve()
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.suffix not in {".json", ".jsonl", ".log", ".txt", ".md", ".yaml"}:
            raise HTTPException(404)
        return FileResponse(p, filename=p.name)

    @app.get("/api/leaderboard")
    def leaderboard():
        rows = []
        for job in db.list():
            if job["kind"] != "benchmark":
                continue
            p = db.root / "runs" / job["id"] / "summary.json"
            if p.exists():
                try:
                    rows.append({"id": job["id"], "name": job["name"], "state": job["state"],
                                 "config": job["config"], **json.loads(p.read_text())})
                except ValueError:
                    continue
        return rows

    @app.get("/api/datasets")
    def datasets():
        rows = []
        for p in (db.root / "datasets").glob("*/manifest.json"):
            d = json.loads(p.read_text())
            rows.append({"id": p.parent.name, **{k: v for k, v in d.items() if k not in {"split_ids", "split_groups"}}})
        return rows

    @app.get("/api/datasets/{did}/preview")
    def preview(did: str):
        if not did.isalnum():
            raise HTTPException(404)
        p = db.root / "datasets" / did / "train.jsonl"
        if not p.exists():
            raise HTTPException(404)
        with p.open() as f:
            rows = [json.loads(line) for _, line in zip(range(3), f)]
        for row in rows:
            for m in row["messages"]:
                if isinstance(m["content"], str):
                    m["content"] = m["content"][:2500]
        return rows

    @app.get("/api/skills")
    def skills():
        return [{"id": p.parent.name, **json.loads(p.read_text())} for p in (db.root / "skills").glob("*/metadata.json")]

    @app.post("/api/skills")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(10*1024*1024+1)
        return import_skill(db.root, file.filename or "skill.zip", data)

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")
    return app
