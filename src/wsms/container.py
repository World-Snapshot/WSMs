"""Single-file, range-readable container for World Snapshot Models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence


MAGIC = b"WSMFILE1"
VERSION = 1
HEADER = struct.Struct("<8sIIQQ32s")
HEADER_SIZE = HEADER.size
DEFAULT_ALIGNMENT = 4096
DEFAULT_FILENAME = "WSM_v1.wsm"
DEFAULT_READ_CHUNK = 16 * 1024 * 1024


def _safe_member_name(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe WSM container member: {value!r}")
    return path.as_posix()


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _copy_and_digest(source: Path, output: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    length = 0
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            length += len(chunk)
    return length, digest.hexdigest()


def build_container(
    output_path: str | Path,
    members: Sequence[tuple[str, str | Path]],
    *,
    model_index: Mapping[str, Any],
    alignment: int = DEFAULT_ALIGNMENT,
) -> dict[str, Any]:
    """Build one uncompressed, seekable WSM checkpoint.

    Every payload receives a byte range in the terminal JSON index. The fixed
    64-byte header points to that index, so readers need only two range reads
    before fetching a selected TunnelingNet or backbone component.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if alignment < HEADER_SIZE or alignment & (alignment - 1):
        raise ValueError("alignment must be a power of two and at least 64")
    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for name, source_value in members:
        safe = _safe_member_name(name)
        if safe in seen:
            raise ValueError(f"Duplicate WSM container member: {safe}")
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        seen.add(safe)
        normalized.append((safe, source))

    temporary = destination.with_suffix(destination.suffix + ".incomplete")
    records: dict[str, dict[str, Any]] = {}
    with temporary.open("w+b") as output:
        output.write(b"\0" * HEADER_SIZE)
        for ordinal, (name, source) in enumerate(normalized, start=1):
            offset = _align(output.tell(), alignment)
            padding = offset - output.tell()
            if padding:
                output.write(b"\0" * padding)
            length, digest = _copy_and_digest(source, output)
            records[name] = {
                "offset": offset,
                "length": length,
                "sha256": digest,
            }
            if ordinal % 100 == 0 or ordinal == len(normalized):
                print(f"Packed {ordinal:,}/{len(normalized):,} members", flush=True)

        index_offset = _align(output.tell(), alignment)
        padding = index_offset - output.tell()
        if padding:
            output.write(b"\0" * padding)
        document = {
            "format": "world-snapshot-model-single-file-v1",
            "version": VERSION,
            "alignment": alignment,
            "model": dict(model_index),
            "members": records,
        }
        index_bytes = json.dumps(
            document, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        output.write(index_bytes)
        output.flush()
        os.fsync(output.fileno())
        output.seek(0)
        output.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                alignment,
                index_offset,
                len(index_bytes),
                hashlib.sha256(index_bytes).digest(),
            )
        )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    return document


class _RangeSource:
    def read(self, offset: int, length: int) -> bytes:
        raise NotImplementedError

    def iter_range(
        self,
        offset: int,
        length: int,
        *,
        chunk_size: int = DEFAULT_READ_CHUNK,
    ) -> Iterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        position = offset
        remaining = length
        while remaining:
            size = min(chunk_size, remaining)
            yield self.read(position, size)
            position += size
            remaining -= size


class _LocalRangeSource(_RangeSource):
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self, offset: int, length: int) -> bytes:
        with self.path.open("rb") as handle:
            handle.seek(offset)
            value = handle.read(length)
        if len(value) != length:
            raise EOFError(
                f"Short read from {self.path}: {len(value)} != {length}"
            )
        return value

    def iter_range(
        self,
        offset: int,
        length: int,
        *,
        chunk_size: int = DEFAULT_READ_CHUNK,
    ) -> Iterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        with self.path.open("rb") as handle:
            handle.seek(offset)
            remaining = length
            while remaining:
                value = handle.read(min(chunk_size, remaining))
                if not value:
                    raise EOFError(f"Short read from {self.path}")
                yield value
                remaining -= len(value)


class _HubRangeSource(_RangeSource):
    def __init__(
        self,
        repo_id: str,
        *,
        filename: str,
        revision: str | None,
        token: str | bool | None,
    ) -> None:
        try:
            import httpx
            from huggingface_hub import get_token, hf_hub_url
        except ImportError as error:  # pragma: no cover - packaging contract
            raise RuntimeError("huggingface_hub and httpx are required") from error
        self._httpx = httpx
        self.url = hf_hub_url(repo_id, filename=filename, revision=revision)
        resolved_token = get_token() if token is None or token is True else token
        self.headers = (
            {"Authorization": f"Bearer {resolved_token}"}
            if isinstance(resolved_token, str) and resolved_token
            else {}
        )

    def read(self, offset: int, length: int) -> bytes:
        headers = dict(self.headers)
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
        with self._httpx.Client(follow_redirects=True, timeout=120.0) as client:
            response = client.get(self.url, headers=headers)
        if response.status_code != 206:
            raise RuntimeError(
                "WSM server did not honor byte-range loading "
                f"({response.status_code}) for {self.url}"
            )
        if len(response.content) != length:
            raise EOFError(
                f"Short HTTP range read: {len(response.content)} != {length}"
            )
        return response.content

    def iter_range(
        self,
        offset: int,
        length: int,
        *,
        chunk_size: int = DEFAULT_READ_CHUNK,
    ) -> Iterator[bytes]:
        # Separate moderate HTTP ranges keep peak memory bounded and allow a
        # failed transfer to be retried by the caller without restarting a
        # multi-gigabyte member from byte zero.
        yield from super().iter_range(offset, length, chunk_size=chunk_size)


@dataclass(frozen=True, slots=True)
class WSMContainer:
    """Opened local or remote monolithic WSM checkpoint."""

    source: _RangeSource
    index: dict[str, Any]
    identity: str

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        filename: str = DEFAULT_FILENAME,
        revision: str | None = None,
        token: str | bool | None = None,
    ) -> "WSMContainer":
        candidate = Path(str(source)).expanduser()
        if candidate.is_dir():
            candidate = candidate / filename
        if candidate.is_file():
            range_source: _RangeSource = _LocalRangeSource(candidate.resolve())
            identity = f"local:{candidate.resolve()}:{candidate.stat().st_mtime_ns}"
        else:
            range_source = _HubRangeSource(
                str(source), filename=filename, revision=revision, token=token
            )
            identity = f"hub:{source}:{revision or 'main'}:{filename}"
        header = range_source.read(0, HEADER_SIZE)
        magic, version, _, index_offset, index_length, expected = HEADER.unpack(header)
        if magic != MAGIC or version != VERSION:
            raise ValueError("Not a supported WSM single-file checkpoint")
        index_bytes = range_source.read(index_offset, index_length)
        if hashlib.sha256(index_bytes).digest() != expected:
            raise ValueError("WSM container index checksum mismatch")
        document = json.loads(index_bytes)
        if document.get("format") != "world-snapshot-model-single-file-v1":
            raise ValueError("Unsupported WSM container index")
        return cls(range_source, document, identity)

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.index["model"])

    def member(self, name: str) -> dict[str, Any]:
        safe = _safe_member_name(name)
        value = self.index.get("members", {}).get(safe)
        if value is None:
            raise KeyError(f"WSM container has no member {safe}")
        return dict(value)

    def read_member(self, name: str) -> bytes:
        record = self.member(name)
        value = self.source.read(int(record["offset"]), int(record["length"]))
        if hashlib.sha256(value).hexdigest() != record["sha256"]:
            raise ValueError(f"WSM member checksum mismatch: {name}")
        return value

    def copy_member_to(
        self,
        name: str,
        output: BinaryIO,
        *,
        chunk_size: int = DEFAULT_READ_CHUNK,
    ) -> None:
        record = self.member(name)
        digest = hashlib.sha256()
        length = 0
        for chunk in self.source.iter_range(
            int(record["offset"]),
            int(record["length"]),
            chunk_size=chunk_size,
        ):
            output.write(chunk)
            digest.update(chunk)
            length += len(chunk)
        if length != int(record["length"]):
            raise EOFError(f"Short WSM member read: {name}")
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"WSM member checksum mismatch: {name}")

    def verify_member(
        self,
        name: str,
        *,
        chunk_size: int = DEFAULT_READ_CHUNK,
    ) -> None:
        record = self.member(name)
        digest = hashlib.sha256()
        length = 0
        for chunk in self.source.iter_range(
            int(record["offset"]),
            int(record["length"]),
            chunk_size=chunk_size,
        ):
            digest.update(chunk)
            length += len(chunk)
        if length != int(record["length"]):
            raise EOFError(f"Short WSM member read: {name}")
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"WSM member checksum mismatch: {name}")

    def verify(self, names: Iterable[str] | None = None) -> int:
        selected = list(names) if names is not None else list(self.index["members"])
        for ordinal, name in enumerate(selected, 1):
            self.verify_member(name)
            if ordinal % 100 == 0 or ordinal == len(selected):
                print(f"Verified {ordinal:,}/{len(selected):,} members", flush=True)
        return len(selected)

    def materialize(
        self,
        names: Iterable[str],
        cache_dir: str | Path | None = None,
    ) -> Path:
        cache = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path.home() / ".cache" / "wsms"
        )
        key = hashlib.sha256(self.identity.encode()).hexdigest()[:20]
        root = cache / "single_file" / key
        for name in names:
            safe = _safe_member_name(name)
            record = self.member(safe)
            destination = root / safe
            marker = destination.with_suffix(destination.suffix + ".sha256")
            if (
                destination.is_file()
                and destination.stat().st_size == int(record["length"])
                and marker.is_file()
                and marker.read_text(encoding="ascii").strip() == record["sha256"]
            ):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=destination.name, delete=False
            ) as handle:
                temporary = Path(handle.name)
                try:
                    self.copy_member_to(safe, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(destination)
            marker.write_text(str(record["sha256"]) + "\n", encoding="ascii")
        return root

    def task_entry(self, identity: str) -> dict[str, Any]:
        matches = [
            dict(value)
            for value in self.model.get("tasks", [])
            if str(value.get("tunnelingnet_id")) == identity
        ]
        if len(matches) != 1:
            raise KeyError(f"WSM checkpoint has {len(matches)} tasks for {identity}")
        return matches[0]
