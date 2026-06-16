"""Source separation (HTDemucs) — split a track into drums/bass/other/vocals.

The separated stems have far higher SNR than the full mix, so re-tagging the
drum/'other' stem lifts quiet percussion (cowbells, shakers) and the vocal stem
makes language ID viable. This is the opt-in "deep analysis" pre-step.

Heavy import (demucs) is LAZY; downloads honor the OS trust store. The real
backend is injectable (``separator=`` callable) so the orchestration is testable.
"""

from __future__ import annotations

import os

_INSTALL_HINT = "Deep pass (stem separation) needs Demucs: pip install demucs"


def separate(path: str, out_dir: str, model_name: str = "htdemucs", separator=None) -> dict:
    """Separate ``path`` into stems -> ``{stem_name: wav_path}``.

    ``separator(path, out_dir) -> {stem: wav}`` overrides the real Demucs backend
    (used in tests). Returns {} on failure.
    """
    os.makedirs(out_dir, exist_ok=True)
    if separator is not None:
        return separator(path, out_dir)
    return _demucs_separate(path, out_dir, model_name)


def _demucs_separate(path: str, out_dir: str, model_name: str) -> dict:
    from mgc._net import enable_os_truststore

    enable_os_truststore()
    try:
        import torch  # type: ignore
        from demucs.apply import apply_model  # type: ignore
        from demucs.audio import AudioFile, save_audio  # type: ignore
        from demucs.pretrained import get_model  # type: ignore
    except Exception as e:  # pragma: no cover - heavy dep missing
        raise RuntimeError(_INSTALL_HINT) from e

    model = get_model(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    wav = AudioFile(path).read(streams=0, samplerate=model.samplerate,
                               channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, wav[None].to(device), device=device, progress=False)[0]
    sources = sources * ref.std() + ref.mean()

    out: dict = {}
    for name, source in zip(model.sources, sources):
        dest = os.path.join(out_dir, f"{name}.wav")
        save_audio(source.cpu(), dest, model.samplerate)
        out[name] = dest
    return out
