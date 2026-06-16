"""Sung-language identification (Whisper) on a separated vocal stem.

This is the *metadata-ish* path to "what language is this sung in" — run on the
isolated vocal stem (accompaniment removed) for much better accuracy. Closed set
of major languages; gate on confidence. Region/dialect is NOT inferred here.

Lazy import; detector is injectable for tests. Prefers openai-whisper's fast
``detect_language`` (no full transcription); falls back to faster-whisper.
"""

from __future__ import annotations

_INSTALL_HINT = ("Deep pass (sung-language ID) needs Whisper: "
                 "pip install openai-whisper  (or faster-whisper)")


def detect_language(path: str, detector=None, model_name: str = "small") -> dict:
    """Detect the (sung) language of an audio file -> ``{language, confidence}``.

    ``detector(path) -> {language, confidence}`` overrides the real backend.
    """
    if detector is not None:
        return detector(path)
    return _whisper_language(path, model_name)


def _whisper_language(path: str, model_name: str) -> dict:
    from mgc._net import enable_os_truststore

    enable_os_truststore()
    # Preferred: openai-whisper detect_language (only needs the first 30 s).
    try:
        import whisper  # type: ignore

        model = whisper.load_model(model_name)
        audio = whisper.pad_or_trim(whisper.load_audio(path))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        return {"language": lang, "confidence": round(float(probs[lang]), 3)}
    except ImportError:
        pass

    # Fallback: faster-whisper (CTranslate2).
    try:
        import torch  # type: ignore
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_name, device=device,
                         compute_type="float16" if device == "cuda" else "int8")
    _segments, info = model.transcribe(path, language=None)
    return {"language": info.language, "confidence": round(float(info.language_probability), 3)}
