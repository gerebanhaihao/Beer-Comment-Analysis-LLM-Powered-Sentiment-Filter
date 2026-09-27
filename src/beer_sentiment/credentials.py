"""Save desktop API keys encrypted for the current Windows user."""

from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path


class DataBlob(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD),
        ("data", ctypes.POINTER(ctypes.c_byte)),
    ]


def credentials_path() -> Path:
    app_data = os.environ.get("APPDATA")
    base = Path(app_data) if app_data else Path.home() / "AppData" / "Roaming"
    return base / "BeerSentiment" / "credentials.json"


def _crypt(value: bytes, *, decrypt: bool) -> bytes:
    source_buffer = ctypes.create_string_buffer(value)
    source = DataBlob(
        len(value), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte))
    )
    result = DataBlob()
    crypt32 = ctypes.windll.crypt32
    if decrypt:
        function = crypt32.CryptUnprotectData
        args = (
            ctypes.byref(source),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(result),
        )
    else:
        function = crypt32.CryptProtectData
        args = (
            ctypes.byref(source),
            "BeerSentiment API key",
            None,
            None,
            None,
            0,
            ctypes.byref(result),
        )
    function.argtypes = [
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p if decrypt else wintypes.LPCWSTR,
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    function.restype = wintypes.BOOL
    if not function(*args):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        local_free = ctypes.windll.kernel32.LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p
        local_free(ctypes.cast(result.data, ctypes.c_void_p))


def load_saved_keys(path: Path | None = None) -> dict[str, str]:
    target = path or credentials_path()
    if not target.exists():
        return {}
    encoded = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(encoded, dict):
        raise TypeError("保存的 API 密钥文件格式错误")
    return {
        model: _crypt(base64.b64decode(value), decrypt=True).decode("utf-8")
        for model, value in encoded.items()
        if isinstance(model, str) and isinstance(value, str)
    }


def save_key(model: str, key: str, path: Path | None = None) -> Path:
    target = path or credentials_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    if not isinstance(encoded, dict):
        raise TypeError("保存的 API 密钥文件格式错误")
    encoded[model] = base64.b64encode(_crypt(key.encode("utf-8"), decrypt=False)).decode(
        "ascii"
    )
    target.write_text(json.dumps(encoded, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
