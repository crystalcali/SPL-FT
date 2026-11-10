import io
import json
from pathlib import Path, PurePosixPath
import uuid
import zipfile
from .data import digest


def import_skill(root: Path, name: str, payload: bytes):
    if len(payload) > 10 * 1024 * 1024:
        raise ValueError("Skill upload exceeds 10 MiB")
    files = {}
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            total = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                p = PurePosixPath(info.filename)
                if p.is_absolute() or ".." in p.parts or "\\" in info.filename or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unsafe archive member")
                total += info.file_size
                if total > 20 * 1024 * 1024 or len(files) >= 200:
                    raise ValueError("Expanded skill archive exceeds limits")
                files[str(p)] = archive.read(info)
    elif name.lower().endswith(".md"):
        files["SKILL.md"] = payload
    else:
        raise ValueError("Upload a .zip directory or a SKILL.md file")
    if not any(Path(p).name == "SKILL.md" for p in files):
        raise ValueError("No SKILL.md found")
    sid = uuid.uuid4().hex[:12]
    folder = root / "skills" / sid
    content = folder / "content"
    for path, data in files.items():
        target = content / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # A zip may have a single wrapper directory; expose that root to Harbor.
    roots = [p.parent for p in content.rglob("SKILL.md")]
    if len(roots) == 1 and roots[0] != content:
        nested = roots[0]
        import shutil
        shutil.move(str(nested), str(folder / "unwrapped"))
        shutil.rmtree(content)
        (folder / "unwrapped").rename(content)
    if not (content / "SKILL.md").exists() and not any(p.is_dir() and (p / "SKILL.md").exists() for p in content.iterdir()):
        import shutil
        shutil.rmtree(folder)
        raise ValueError("Skills must be at the archive root or one directory below it")
    delivered = {str(p.relative_to(content)): p.read_bytes().hex() for p in sorted(content.rglob("*")) if p.is_file()}
    meta = {"name": name, "sha256": digest(delivered), "files": len(delivered)}
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"id": sid, **meta}
