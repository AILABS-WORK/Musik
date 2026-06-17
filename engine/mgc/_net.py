"""Networking helper: optionally route TLS through the OS trust store.

Some environments (corporate proxies, TLS inspection) present a CA that
OpenSSL 3 rejects (e.g. "Basic Constraints of CA cert not marked critical"),
which breaks Hugging Face model downloads. ``truststore`` makes Python verify
via the OS cert store instead, which trusts that CA. This is a no-op when
truststore isn't installed.
"""

from __future__ import annotations

import os

_DONE = False

# Hugging Face's accelerated download backends (hf_transfer and the Xet protocol)
# hang on large weight files behind TLS-inspecting proxies — the small config
# files download, then the big .safetensors stalls at 0 bytes. Force the plain
# requests path (which truststore patches), which downloads reliably. Set as early
# as possible (import time) and honor any value the user already set.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def enable_os_truststore() -> bool:
    """Inject the OS trust store into Python's TLS (once). Returns True if active.

    Also pins the reliable HF download backend (see module note) before any model
    download runs.
    """
    global _DONE
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if _DONE:
        return True
    try:
        import truststore

        truststore.inject_into_ssl()
        _DONE = True
        return True
    except Exception:
        return False
