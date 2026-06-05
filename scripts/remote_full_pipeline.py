from __future__ import annotations

import argparse
import os
import posixpath
import shutil
import stat
import tarfile
import time
from pathlib import Path

import paramiko


CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent if CODE_ROOT.name in {"codes", "published_codes"} else CODE_ROOT
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "remote_full_pipeline.log"


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_account() -> tuple[str, str, str, int]:
    tokens = (ROOT / "ssh-account.txt").read_text(encoding="utf-8").strip().split()
    if tokens[0].lower() == "ssh":
        user, host = tokens[1].split("@", 1)
        port = 22
        password = ""
        i = 2
        while i < len(tokens):
            if tokens[i] in {"-p", "-P"} and i + 1 < len(tokens):
                port = int(tokens[i + 1])
                i += 2
            else:
                password = tokens[i].split(":", 1)[-1]
                i += 1
        return host, user, password, port
    if "@" in tokens[0]:
        user, host = tokens[0].split("@", 1)
        return host, user, tokens[1].split(":", 1)[-1], 22
    host, user, password = tokens[:3]
    port = int(tokens[3]) if len(tokens) > 3 and tokens[3].isdigit() else 22
    return host, user, password.split(":", 1)[-1], port


def connect() -> paramiko.SSHClient:
    host, user, password, port = parse_account()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=password, port=port, timeout=30)
    return client


def remote_stat(sftp: paramiko.SFTPClient, path: str):
    try:
        return sftp.stat(path)
    except FileNotFoundError:
        return None
    except OSError:
        return None


def mkdir_p(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = []
    cur = path
    while cur not in {"", "/"}:
        parts.append(cur)
        cur = posixpath.dirname(cur)
    for part in reversed(parts):
        try:
            sftp.mkdir(part)
        except OSError:
            pass


def iter_files() -> list[Path]:
    roots = [
        CODE_ROOT / "configs",
        CODE_ROOT / "scripts",
        CODE_ROOT / "src",
        ROOT / "data" / "raw",
        ROOT / "data" / "external",
    ]
    top_files = [
        CODE_ROOT / "pyproject.toml",
        CODE_ROOT / "README.md",
        CODE_ROOT / "research_plan.md",
        CODE_ROOT / "requirements-remote-minimal.txt",
    ]
    files: list[Path] = [p for p in top_files if p.exists()]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(CODE_ROOT) if path.is_relative_to(CODE_ROOT) else path.relative_to(ROOT)
            if path.is_relative_to(CODE_ROOT):
                rel = Path("codes") / rel
            if "__pycache__" in rel.parts:
                continue
            if rel.parts[:3] == ("data", "raw", "fetal_kidney_hca"):
                continue
            if path.suffix in {".pyc", ".part"}:
                continue
            files.append(path)
    return sorted(files)


def upload_tree(client: paramiko.SSHClient, remote_root: str) -> None:
    sftp = client.open_sftp()
    files = iter_files()
    total = sum(p.stat().st_size for p in files)
    uploaded = 0
    skipped = 0
    log(f"sync_start files={len(files)} bytes={total}")
    for path in files:
        rel_path = path.relative_to(CODE_ROOT) if path.is_relative_to(CODE_ROOT) else path.relative_to(ROOT)
        if path.is_relative_to(CODE_ROOT):
            rel_path = Path("codes") / rel_path
        rel = rel_path.as_posix()
        remote_path = posixpath.join(remote_root, rel)
        st = remote_stat(sftp, remote_path)
        size = path.stat().st_size
        if st and stat.S_ISREG(st.st_mode) and st.st_size == size:
            skipped += size
            continue
        mkdir_p(sftp, posixpath.dirname(remote_path))
        tmp_remote = remote_path + ".uploading"
        log(f"upload {rel} size={size}")
        last_report = time.time()

        def progress(done: int, total_bytes: int) -> None:
            nonlocal last_report
            now = time.time()
            if now - last_report >= 30 or done == total_bytes:
                pct = 100.0 * done / max(1, total_bytes)
                log(f"upload_progress {rel} {pct:.1f}% {done}/{total_bytes}")
                last_report = now

        sftp.put(str(path), tmp_remote, callback=progress)
        try:
            sftp.remove(remote_path)
        except OSError:
            pass
        sftp.rename(tmp_remote, remote_path)
        uploaded += size
    sftp.close()
    log(f"sync_complete uploaded={uploaded} skipped={skipped}")


def run_remote(client: paramiko.SSHClient, remote_root: str) -> int:
    command = (
        f"cd {remote_root} && mkdir -p logs .cache/tmp && "
        "bash codes/scripts/remote_run.sh 2>&1 | tee logs/full_remote_run.log; "
        "status=${PIPESTATUS[0]}; "
        "tar -czf wt-ai-full-results.tar.gz results logs codes; "
        "exit $status"
    )
    log("remote_run_start")
    stdin, stdout, stderr = client.exec_command(f"bash -lc {quote(command)}", get_pty=False)
    for line in iter(stdout.readline, ""):
        log("REMOTE " + line.rstrip())
    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        log("REMOTE_STDERR " + err.strip())
    status = stdout.channel.recv_exit_status()
    log(f"remote_run_exit status={status}")
    return status


def quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def pull_results(client: paramiko.SSHClient, remote_root: str) -> None:
    out = ROOT / "remote_artifacts_full"
    out.mkdir(exist_ok=True)
    local_tar = out / "wt-ai-full-results.tar.gz"
    sftp = client.open_sftp()
    remote_tar = posixpath.join(remote_root, "wt-ai-full-results.tar.gz")
    log("pull_results_start")
    sftp.get(remote_tar, str(local_tar))
    sftp.close()
    extract_dir = out / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(extract_dir)
    log(f"pull_results_complete {local_tar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    client = connect()
    stdin, stdout, stderr = client.exec_command("printf %s \"$HOME\"")
    home = stdout.read().decode("utf-8")
    remote_root = posixpath.join(home, "wt-ai")
    client.exec_command(f"mkdir -p {quote(remote_root)}")[1].channel.recv_exit_status()
    client.exec_command(f"find {quote(remote_root)} -name '*.uploading' -type f -delete")[1].channel.recv_exit_status()
    try:
        if not args.skip_sync:
            upload_tree(client, remote_root)
        status = 0
        if not args.skip_run:
            status = run_remote(client, remote_root)
        pull_results(client, remote_root)
        if status != 0:
            raise SystemExit(status)
    finally:
        client.close()


if __name__ == "__main__":
    main()
