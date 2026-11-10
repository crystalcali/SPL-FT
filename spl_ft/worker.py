import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import threading
from .store import Store


class Worker:
    def __init__(self, store=None):
        self.store = store or Store()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=16)

    def loop(self):
        with open(self.store.root / "worker.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            self.store.recover()
            while not self.stop_event.is_set():
                job = self.store.claim()
                if not job:
                    self.stop_event.wait(1)
                    continue
                self.execute(job)

    def execute(self, job):
        folder = self.store.run_dir(job["id"])
        (folder / "config.json").write_text(json.dumps(job["config"], ensure_ascii=False, indent=2))
        cmd = [sys.executable, "-u", "-m", "spl_ft.tasks", job["kind"], str(folder)]
        env = {**os.environ, "SPL_FT_HOME": str(self.store.root), "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
        try:
            with open(folder / "console.log", "a", buffering=1) as log:
                log.write(f"Starting {job['kind']} at {time.ctime()}\n")
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                self.store.update(job["id"], pid=proc.pid)
                cancelled = False
                while proc.poll() is None:
                    if self.stop_event.is_set() or self.store.get(job["id"])["cancel"]:
                        cancelled = True
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        break
                    self.stop_event.wait(0.5)
                state = "interrupted" if self.stop_event.is_set() else "cancelled" if cancelled else "succeeded" if proc.returncode == 0 else "failed"
                self.store.update(job["id"], state=state, finished=time.time(), pid=None,
                                  error=None if state == "succeeded" else f"Process ended ({proc.returncode}); see console log")
        except Exception as e:
            self.store.update(job["id"], state="failed", finished=time.time(), pid=None, error=str(e))
