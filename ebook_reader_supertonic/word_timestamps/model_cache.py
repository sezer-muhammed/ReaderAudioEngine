from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


MODEL_NAME_EN_US_LGRAPH_0_22 = "vosk-model-en-us-0.22-lgraph"
MODEL_LANG_EN_US = "en-us"


HF_REPO_ID = "alphacep/vosk-models"
HF_FILENAME_EN_US_LGRAPH_0_22 = f"{MODEL_NAME_EN_US_LGRAPH_0_22}.tar.gz"

OFFICIAL_BASE_URL = "https://alphacephei.com/vosk/models/"


class VoskModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoskModelSpec:
    name: str
    lang_dir: str
    hf_repo_id: str
    hf_filename: str
    official_url: str
    sha256: Optional[str] = None

    def cache_dir(self, cache_base: Path) -> Path:
        return cache_base / "models" / self.lang_dir / self.name


EN_US_LGRAPH_0_22 = VoskModelSpec(
    name=MODEL_NAME_EN_US_LGRAPH_0_22,
    lang_dir=MODEL_LANG_EN_US,
    hf_repo_id=HF_REPO_ID,
    hf_filename=HF_FILENAME_EN_US_LGRAPH_0_22,
    official_url=f"{OFFICIAL_BASE_URL}{MODEL_NAME_EN_US_LGRAPH_0_22}.tar.gz",
    sha256=None,
)


def _vosk_cache_base() -> Path:
    return Path(os.environ.get("VOSK_CACHE_DIR", Path.home() / ".cache" / "vosk"))


def _offline_mode() -> bool:
    return os.environ.get("VOSK_OFFLINE", "0") == "1"


def _is_valid_model_dir(model_dir: Path) -> bool:
    # A minimal sanity check for Vosk model layout.
    return (
        model_dir.is_dir()
        and (model_dir / "am").is_dir()
        and (model_dir / "conf").is_dir()
        and (model_dir / "graph").is_dir()
    )


def _acquire_lock(lock_path: Path, timeout_s: float = 600.0, poll_s: float = 0.2) -> int:
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd
        except FileExistsError:
            if time.time() >= deadline:
                raise VoskModelError(f"Timeout waiting for lock: {lock_path}")
            time.sleep(poll_s)


def _release_lock(lock_path: Path, fd: Optional[int]) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except Exception:
            pass
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        # Best effort: lock file cleanup shouldn't break the caller.
        pass


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_tar_members(tar: tarfile.TarFile, extract_to: Path) -> Iterable[tarfile.TarInfo]:
    base = extract_to.resolve()
    for member in tar.getmembers():
        member_path = (extract_to / member.name).resolve()
        if base not in member_path.parents and member_path != base:
            raise VoskModelError(f"Unsafe path in tar archive: {member.name}")
        yield member


def _download_from_hf(spec: VoskModelSpec, target: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:
        raise VoskModelError("huggingface-hub is required for HuggingFace downloads") from e

    downloaded = hf_hub_download(repo_id=spec.hf_repo_id, filename=spec.hf_filename)
    import shutil

    shutil.copyfile(downloaded, target)


def _download_from_official(url: str, target: Path) -> None:
    import urllib.request

    with urllib.request.urlopen(url) as resp, target.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _extract_tar_gz(archive_path: Path, extract_parent: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_parent, members=_safe_tar_members(tar, extract_parent))


def ensure_vosk_model(spec: VoskModelSpec) -> Path:
    """
    Ensure the given Vosk model exists on disk (download + extract once).

    Cache layout (HF-style):
      ~/.cache/vosk/models/<lang>/<model_name>/
    """
    cache_base = _vosk_cache_base()
    model_dir = spec.cache_dir(cache_base)
    if _is_valid_model_dir(model_dir):
        return model_dir

    if _offline_mode():
        raise VoskModelError(
            f"Offline mode (VOSK_OFFLINE=1) and model not found: {model_dir}"
        )

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir.parent / f".{spec.name}.lock"

    lock_fd: Optional[int] = None
    lock_fd = _acquire_lock(lock_path)
    try:
        if _is_valid_model_dir(model_dir):
            return model_dir

        with tempfile.TemporaryDirectory(prefix=f"vosk-{spec.name}-") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            archive_path = tmp_dir_path / spec.hf_filename

            hf_err: Optional[BaseException] = None
            official_err: Optional[BaseException] = None

            try:
                _download_from_hf(spec, archive_path)
            except BaseException as e:
                hf_err = e
                try:
                    _download_from_official(spec.official_url, archive_path)
                except BaseException as e2:
                    official_err = e2

            if not archive_path.exists():
                raise VoskModelError(
                    f"Failed to download {spec.name} (HF={hf_err!r}, official={official_err!r})"
                )

            if spec.sha256:
                actual = _sha256_file(archive_path)
                if actual.lower() != spec.sha256.lower():
                    raise VoskModelError(
                        f"SHA256 mismatch for {spec.name}: expected {spec.sha256}, got {actual}"
                    )

            extract_parent = tmp_dir_path / "extract"
            extract_parent.mkdir(parents=True, exist_ok=True)
            _extract_tar_gz(archive_path, extract_parent)

            extracted = extract_parent / spec.name
            if not extracted.exists():
                # Some archives may include a leading folder or different root naming.
                candidates = [p for p in extract_parent.iterdir() if p.is_dir()]
                if len(candidates) == 1:
                    extracted = candidates[0]
            if not _is_valid_model_dir(extracted):
                raise VoskModelError(f"Extracted model layout invalid: {extracted}")

            # Atomic-ish move into final cache dir.
            if model_dir.exists():
                # Old/partial data: remove only after we have a valid extraction.
                for child in model_dir.iterdir():
                    if child.is_dir():
                        import shutil

                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except Exception:
                            pass
                try:
                    model_dir.rmdir()
                except Exception:
                    pass

            extracted.replace(model_dir)

        if not _is_valid_model_dir(model_dir):
            raise VoskModelError(f"Model directory missing after install: {model_dir}")
        return model_dir
    finally:
        _release_lock(lock_path, lock_fd)


def ensure_vosk_en_us_lgraph_0_22() -> Path:
    return ensure_vosk_model(EN_US_LGRAPH_0_22)


def default_vosk_model_for_lang(lang: Optional[str]) -> Optional[VoskModelSpec]:
    if lang is None:
        return EN_US_LGRAPH_0_22
    lang = lang.lower()
    if lang.startswith("en"):
        return EN_US_LGRAPH_0_22
    return None
