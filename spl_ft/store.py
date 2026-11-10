"""Durable queue. One worker owns a file lock; claims are transactional."""
import json
import sqlite3
import time
import uuid
from pathlib import Path
from .config import home


class Store:
    def __init__(self, root: Path | None = None):
        self.root = root or home()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
              config TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
              priority INTEGER NOT NULL DEFAULT 0, dependency TEXT,
              created REAL NOT NULL, started REAL, finished REAL,
              cancel INTEGER NOT NULL DEFAULT 0, error TEXT, pid INTEGER);
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def enqueue(self, kind, config, dependency=None, priority=0):
        jid = uuid.uuid4().hex[:12]
        with self.connect() as db:
            if dependency and not db.execute("SELECT 1 FROM jobs WHERE id=?", (dependency,)).fetchone():
                raise ValueError("Unknown dependency")
            db.execute("INSERT INTO jobs(id,kind,name,config,priority,dependency,created) VALUES(?,?,?,?,?,?,?)",
                       (jid, kind, config.get("name", kind), json.dumps(config), priority, dependency, time.time()))
        return self.get(jid)

    def get(self, jid):
        with self.connect() as db:
            r = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not r:
            raise KeyError(jid)
        return self.decode(r)

    @staticmethod
    def decode(r):
        d = dict(r)
        d["config"] = json.loads(d["config"])
        return d

    def list(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC")]

    def update(self, jid, **values):
        allowed = {"state", "priority", "cancel", "error", "pid", "started", "finished"}
        if not values or not set(values) <= allowed:
            raise ValueError("Invalid queue update")
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in values) + " WHERE id=?", (*values.values(), jid))

    def setting(self, key, default=None):
        with self.connect() as db:
            r = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_setting(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, json.dumps(value)))

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""UPDATE jobs SET state='blocked',error='Dependency did not succeed',finished=?
              WHERE state='queued' AND dependency IN
              (SELECT id FROM jobs WHERE state IN ('failed','cancelled','interrupted','blocked'))""", (time.time(),))
            if self.setting("paused", False):
                return None
            r = db.execute("""SELECT j.* FROM jobs j LEFT JOIN jobs p ON p.id=j.dependency
              WHERE j.state='queued' AND j.cancel=0 AND (j.dependency IS NULL OR p.state='succeeded')
              ORDER BY j.priority DESC,j.created LIMIT 1""").fetchone()
            if not r:
                return None
            db.execute("UPDATE jobs SET state='running',started=? WHERE id=?", (time.time(), r["id"]))
            return {**self.decode(r), "state": "running"}

    def recover(self):
        # Clean only verified orphan task process groups from this workspace.
        # A reused PID or an unrelated user model server must never be killed.
        import os
        import signal
        import psutil
        for job in self.list():
            if job["state"] != "running" or not job["pid"]:
                continue
            try:
                proc = psutil.Process(job["pid"])
                cmd = proc.cmdline()
                expected = str(self.root / "runs" / job["id"])
                if "spl_ft.tasks" in cmd and expected in cmd and os.getpgid(proc.pid) == proc.pid:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=6)
                    except psutil.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
            except (psutil.NoSuchProcess, ProcessLookupError, PermissionError):
                pass
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='interrupted',error='Worker stopped; retry or resume from checkpoint',finished=? WHERE state='running'", (time.time(),))

    def run_dir(self, jid):
        self.get(jid)
        p = self.root / "runs" / jid
        p.mkdir(parents=True, exist_ok=True)
        return p
