import gzip
import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _update_hash(h: "hashlib._Hash", obj: Any) -> None:
    if obj is None:
        h.update(b"n")
        return
    if isinstance(obj, bool):
        h.update(b"b1" if obj else b"b0")
        return
    if isinstance(obj, int):
        h.update(b"i")
        h.update(str(obj).encode("utf-8"))
        h.update(b";")
        return
    if isinstance(obj, float):
        h.update(b"f")
        h.update(repr(obj).encode("utf-8"))
        h.update(b";")
        return
    if isinstance(obj, str):
        h.update(b"s")
        h.update(obj.encode("utf-8"))
        h.update(b";")
        return
    if isinstance(obj, bytes):
        h.update(b"y")
        h.update(obj)
        h.update(b";")
        return
    if isinstance(obj, (list, tuple)):
        h.update(b"[")
        for x in obj:
            _update_hash(h, x)
            h.update(b",")
        h.update(b"]")
        return
    if isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj.keys(), key=lambda x: (str(type(x)), repr(x))):
            _update_hash(h, k)
            h.update(b":")
            _update_hash(h, obj[k])
            h.update(b",")
        h.update(b"}")
        return

    h.update(b"r")
    h.update(repr(obj).encode("utf-8"))
    h.update(b";")


def make_hash(obj: Any) -> str:
    h = hashlib.sha256()
    _update_hash(h, obj)
    return h.hexdigest()


@dataclass
class DiskCache:
    cache_dir: Path
    enabled: bool = True
    refresh: bool = False

    def _path(self, namespace: str, key: str) -> Path:
        subdir = self.cache_dir / namespace / key[:2] / key[2:4]
        return subdir / f"{key}.pkl.gz"

    def get(self, namespace: str, key_obj: Any) -> Optional[Any]:
        if not self.enabled or self.refresh:
            return None
        key = make_hash(key_obj)
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, namespace: str, key_obj: Any, value: Any) -> None:
        if not self.enabled:
            return
        key = make_hash(key_obj)
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
