from __future__ import annotations

import os
import posixpath
import stat
import shlex
import time
from pathlib import Path

import paramiko


CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent if CODE_ROOT.name in {"codes", "published_codes"} else CODE_ROOT
LOG = ROOT / "logs" / "remote_assets_upload.log"
LOG.parent.mkdir(exist_ok=True)


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_account() -> tuple[str, str, str, int]:
    tokens = (ROOT / "ssh-account.txt").read_text(encoding="utf-8").strip().split()
    user, host = tokens[1].split("@", 1)
    port = int(tokens[3])
    password = tokens[4].split(":", 1)[-1]
    return host, user, password, port


def connect() -> paramiko.SSHClient:
    host, user, password, port = parse_account()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=password, port=port, timeout=30)
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(30)
    return client


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


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int | None:
    try:
        st = sftp.stat(path)
    except OSError:
        return None
    return st.st_size if stat.S_ISREG(st.st_mode) else None


def iter_asset_files() -> list[Path]:
    roots = [
        ROOT / "models" / "structure_predictors",
        ROOT / "data" / "raw" / "fetal_kidney_hca",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix != ".part":
                rel = path.relative_to(ROOT)
                if rel.parts[:4] == ("models", "structure_predictors", "boltz", "mols"):
                    continue
                if rel.parts[:4] == ("models", "structure_predictors", "alphafold_multimer", "params"):
                    continue
                files.append(path)
    return sorted(files)


def remote_extract_archives(client: paramiko.SSHClient, remote_root: str) -> None:
    command = f"""
set -euo pipefail
cd {shlex.quote(remote_root)}
python3 - <<'PY'
from pathlib import Path
import tarfile, time
root = Path.cwd() / "models" / "structure_predictors"
boltz = root / "boltz" / "mols.tar"
if boltz.exists():
    marker = root / "boltz" / "mols" / ".extracted_from_mols_tar"
    if not marker.exists():
        with tarfile.open(boltz) as tar:
            tar.extractall(root / "boltz")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
af = root / "alphafold_multimer" / "alphafold_params_2022-12-06.tar"
if af.exists():
    out = root / "alphafold_multimer" / "params"
    marker = out / ".extracted_from_alphafold_params"
    if not marker.exists():
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(af) as tar:
            tar.extractall(out)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
PY
"""
    log("remote_extract_start")
    stdin, stdout, stderr = client.exec_command(command, timeout=3600)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if out:
        log("remote_extract_stdout " + out)
    if err:
        log("remote_extract_stderr " + err)
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"remote extract failed with status {status}")
    log("remote_extract_complete")


def main() -> None:
    def open_session() -> tuple[paramiko.SSHClient, paramiko.SFTPClient, str]:
        client = connect()
        stdin, stdout, stderr = client.exec_command("printf %s \"$HOME\"")
        home = stdout.read().decode("utf-8")
        return client, client.open_sftp(), posixpath.join(home, "wt-ai")

    client, sftp, remote_root = open_session()
    files = iter_asset_files()
    total = sum(p.stat().st_size for p in files)
    log(f"asset_sync_start files={len(files)} bytes={total}")
    uploaded = 0
    skipped = 0
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        remote_path = posixpath.join(remote_root, rel)
        size = path.stat().st_size
        if remote_size(sftp, remote_path) == size:
            skipped += size
            continue
        mkdir_p(sftp, posixpath.dirname(remote_path))
        tmp = remote_path + ".uploading"
        last = time.time()
        log(f"upload_asset {rel} size={size}")

        def progress(done: int, total_bytes: int) -> None:
            nonlocal last
            now = time.time()
            if now - last >= 30 or done == total_bytes:
                log(f"asset_progress {rel} {100.0 * done / max(1, total_bytes):.1f}% {done}/{total_bytes}")
                last = now

        for attempt in range(1, 6):
            try:
                remote_done = remote_size(sftp, tmp) or 0
                if remote_done > size:
                    sftp.remove(tmp)
                    remote_done = 0
                if remote_done:
                    log(f"upload_resume {rel} offset={remote_done}")
                with path.open("rb") as local_handle:
                    local_handle.seek(remote_done)
                    mode = "r+b" if remote_done else "wb"
                    with sftp.open(tmp, mode) as remote_handle:
                        remote_handle.set_pipelined(True)
                        remote_handle.seek(remote_done)
                        done = remote_done
                        while True:
                            chunk = local_handle.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            remote_handle.write(chunk)
                            done += len(chunk)
                            progress(done, size)
                break
            except Exception as exc:
                log(f"upload_retry {rel} attempt={attempt} error={type(exc).__name__}: {exc}")
                try:
                    sftp.close()
                except Exception:
                    pass
                try:
                    client.close()
                except Exception:
                    pass
                if attempt == 5:
                    raise
                time.sleep(30 * attempt)
                client, sftp, remote_root = open_session()
                remote_path = posixpath.join(remote_root, rel)
                tmp = remote_path + ".uploading"
                mkdir_p(sftp, posixpath.dirname(remote_path))
                if remote_size(sftp, remote_path) == size:
                    break
        try:
            sftp.remove(remote_path)
        except OSError:
            pass
        sftp.rename(tmp, remote_path)
        uploaded += size
    remote_extract_archives(client, remote_root)
    marker = posixpath.join(remote_root, ".assets_ready")
    with sftp.open(marker, "w") as handle:
        handle.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    sftp.close()
    client.close()
    log(f"asset_sync_complete uploaded={uploaded} skipped={skipped} marker={marker}")


if __name__ == "__main__":
    main()
