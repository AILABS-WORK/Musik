"""Networking helper: optionally route TLS through the OS trust store.

Some environments (corporate proxies, TLS inspection) present a CA that
OpenSSL 3 rejects (e.g. "Basic Constraints of CA cert not marked critical"),
which breaks Hugging Face model downloads. ``truststore`` makes Python verify
via the OS cert store instead, which trusts that CA. This is a no-op when
truststore isn't installed.
"""

from __future__ import annotations

_DONE = False


def enable_os_truststore() -> bool:
    """Inject the OS trust store into Python's TLS (once). Returns True if active."""
    global _DONE
    if _DONE:
        return True
    try:
        import truststore

        truststore.inject_into_ssl()
        _DONE = True
        return True
    except Exception:
        return False
