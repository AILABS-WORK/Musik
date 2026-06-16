# Design Spec — Music Understanding & Open-Vocabulary Search

- **Date:** 2026-06-16
- **Status:** Designed (research-grounded), ready to build in phases
- **Builds on:** the existing Musik engine (MERT/CLAP embeddings, by-example classify, BPM/key/energy analysis, set-builder).

> Grounded in a literature sweep of 2022–2026 MIR work (MERT, MuQ/MARBLE, LAION-CLAP, EfficientAT/AST, HTDemucs, Essentia, OpenMIC, VocalSet, Whisper, ReCLAP). Citations at the end.

## 1. Goal
Give every track a rich, honest **"understanding" record** (instruments, vocal/instrumental + gender, mood, optional language/technique, a natural-language caption, BPM/key/energy) and an **open-vocabulary attribute search** ("give me songs with cowbells", "female Spanish vocals, instrumental warmup") — and feed both into an attribute-aware set-builder. Everything runs **locally on the RTX 5080**, offline.

## 2. What we keep (already the modern stack)
Musik already runs the 2024–2026 recipe: **frozen self-supervised music embeddings (MERT-v1-330M) + a small probe head**, an optional **contrastive audio–text model (LAION-CLAP)** for zero-shot/text, and BPM/key/energy. We extend around this rather than replacing it.

## 3. New components (per capability — decided, with honesty about feasibility)

| Capability | Approach | Model / method | Feasibility (local) |
|---|---|---|---|
| **Rich tags** (527-class panel incl. a literal *Cowbell* class) | Supervised AudioSet tagger, once per track at ingest; store the full 527-d probability vector | **EfficientAT `mn40_as`** (default, MIT, ~0.48 mAP, tiny) · **AST** (`MIT/ast-finetuned-audioset`) fallback · PANNs CNN14 baseline | **Solid** |
| **Instruments present** (common, reliable chips) | (a) AudioSet tagger classes + (b) a non-linear **MLP probe on MERT** trained on OpenMIC-2018 | EfficientAT/AST + MERT-features MLP | **Solid** for ~20–50 instruments |
| **Fine percussion / long-tail** ("cowbells", "808s") | Open-vocab CLAP cosine, optionally on a separated stem; treat as soft/percentile scores | LAION-CLAP (+ optional HTDemucs stem) | Open-vocab **solid**; quiet-aux-perc on full mixes **research-grade** (gate behind separation) |
| **Vocal vs instrumental + gender** | Two extra heads on the **Discogs-EffNet** embedding we already wrap | Essentia `voice_instrumental` + `gender` heads | **Solid**, ~free (reuses an existing embedding) |
| **Singing technique** (belt/breathy/vibrato) | MLP probe on MERT/MuQ over the **separated vocal stem** | VocalSet-trained head | **Good but indicative** |
| **Voice register hint** | Coarse low/mid/high from sung-F0 on the vocal stem — **never SATB/fach** | CREPE/torchcrepe F0 | **Speculative** → low-confidence hint only |
| **Sung-language ID** (closed set of major langs) | Language-detect on the **separated vocal stem** | Whisper large-v3 (faster-whisper) | **Good** for major languages |
| **Region / origin of the artist** | **identify the track → look up artist metadata** (NOT from acoustics) | AcoustID/Chromaprint or AudD/ACRCloud → **MusicBrainz**/Discogs artist `area`/`begin-area` | **Solid once the track is identified** (acoustic-only region/dialect remains unsolved) |
| **Mood / emotion** | AudioSet mood classes + Essentia mood/theme + arousal-valence heads + MERT MLP; CLAP for free-text | Essentia + MERT probe | **Solid–good**; gives a 2-D arousal/valence coordinate |
| **Per-song description** | LLM-over-tags from the structured record (and/or a music captioner) | templated/LLM (LP-MusicCaps optional) | **Solid** via templating |
| **Open-vocab search** ("cowbells") | Precomputed CLAP audio vectors (mean **and** per-chunk-max) + prompt-ensemble + **per-query calibration** + a router | LAION-CLAP + ReCLAP prompts | **Solid**, fully local, sub-second queries |

## 4. The novel technique (what ties it together)
1. **LLM-over-tags "understanding compiler"** — fuse the AudioSet-527 vector + OpenMIC instrument probe + Essentia voice/gender + MERT mood probe + (deep) CREPE/Whisper facts + BPM/key/energy into one structured record, then let a local/templated LLM **normalize to canonical tags AND write the caption**. Tags ground the LLM (no inventing instruments); the LLM resolves model conflicts. This connective tissue is what no single model provides.
2. **Hybrid router** — known-taxonomy queries → calibrated AudioSet tagger (precise, thresholdable); free text → CLAP cosine with per-query calibration. Precision on common attributes, universality on the long tail, one ranked output.
3. **Mean + per-chunk-MAX dual CLAP index** — store both so "*contains* a cowbell somewhere" (max over chunks) differs from "cowbell-*driven* throughout" (mean). One extra 512-d vector.
4. **Separation-gated deep analysis** — HTDemucs OFF by default; when toggled, re-run the tagger/CLAP/technique/language models on the isolated **drum/'other'** and **vocal** stems (same models, far higher SNR). The accuracy multiplier, paid only on demand.
5. **Layer-aware MERT probing** — train per-attribute MLP heads on a learned weighted sum of MERT layers (low/mid = timbre/instruments, late = genre/mood) instead of the current mean-over-all-layers vector.
6. **Per-query calibration** — convert raw CLAP cosine into a per-query distribution score (z-score/softmax over the library, or positive-minus-negative-prompt margin) so the UI can offer "return **ALL** matches above X", not just top-k.

## 5. Data model (extends SQLite, non-breaking)
- `embeddings` (exists): add rows `clap` (mean, exists), **`clap_chunkmax`** (new), `discogs_effnet` (exists wrapper) per track.
- **New `understanding` table** (one row/track): `audioset BLOB(float32[527])`, `audioset_model`, `instruments JSON`, `vocal JSON {voice_instrumental,gender,gender_conf,technique,register_hint,language,language_conf}`, `mood JSON {arousal,valence,tags,scores}`, `caption TEXT`, `tags_canonical JSON`, `deep_done INT`, `updated_at`. (Pragmatic fallback: stuff into the existing `analysis.extra` JSON; a dedicated table is cleaner for faceted filtering.)
- Optional `stems(track_id, stem_name, path)` cache so HTDemucs output is reused.

## 6. Flow (two tiers, mirroring the existing light/heavy split)
**Ingest (fast, every track, mostly reuses existing compute):**
decode → BPM/key/energy → MERT embed + CLAP mean + **CLAP chunk-max** → Discogs-EffNet embed → Essentia voice/gender heads → AudioSet tagger (527-d) → MERT MLP probes (instruments=OpenMIC, mood=MTG-Jamendo) → assemble `understanding` (deep_done=0) → LLM-over-tags: canonical tags + caption.

**Deep pass (user-triggered "deep analysis", GPU, cached):**
HTDemucs separate → cache stems → drum/'other' stem: re-run tagger + CLAP for fine percussion → upgrade chips; vocal stem: MERT technique probe (VocalSet) + CREPE register hint + Whisper language → refresh record (deep_done=1), regenerate caption.

**Open-vocab search (query time):**
parse → router: known label → rank by calibrated tagger score (chunk-max for "contains"); else → CLAP text-encoder prompt-ensemble, cosine-rank cached audio vectors, z-score/softmax calibrate → thresholdable. Optionally blend tagger+CLAP. Returns ranked `track_ids` + per-track score.

**Set-builder fusion (upgrade `setbuilder/builder.py`):**
the free-text vibe is ALSO parsed (LLM-over-tags) into **attribute constraints** ("female-vocal peak-time house, instrumental warmup, English-only") that filter the candidate pool via the `understanding` record + CLAP search; keep the energy/BPM arc for ordering, add **Camelot key-compatibility** and **arousal/valence continuity** tie-breakers; emit per-track "reasons" from the tags.

## 7. UI/UX direction (sleek electronic vibe)
Dark glassy "studio at night" — near-black charcoal (#0B0D12), translucent blurred panels, a **neon accent that shifts per song** from its mood (arousal/valence) coordinate; monospaced numerals for BPM/key; restrained reactive motion (waveform pulses with playback).
- **Home/Library:** a prominent **open-vocab search bar** (placeholder cycles real examples: "songs with cowbells", "female Spanish vocals, instrumental warmup", "dark melodic techno ~124 bpm") with a live **router-hint chip** ("matched AudioSet class: Cowbell" vs "open-vocab CLAP") and a **confidence-threshold slider** ("return ALL matches"). Rows show mini chips (instruments, vocal/instrumental, key, BPM, mood dot) + a 1-line caption. Left **facet rail**: vocal/instrumental, gender, language, **Camelot wheel**, BPM range, energy, mood quadrant, instruments.
- **Per-song detail:** hero **interactive waveform** over a **spectrum** strip; an **Understanding** panel of cards (instruments chips opacity-scaled by confidence with a "stem-verified" badge when deep; voice meter + gender confidence ring + language flag + a clearly-muted low-confidence register pill; an **arousal/valence pad** with the song plotted; stat tiles; the caption as a quoted block with regenerate; a **radial tag-cloud** where clicking a tag launches that attribute search).
- **Visualizations:** a library-wide **similarity-map constellation** (UMAP of the chosen embedding, colored by mood, sized by energy, lasso-to-seed-a-set); per-song waveform/spectrum.
- **Set-builder timeline:** a horizontal lane showing the energy **arc** as a glowing curve; each track a card pinned to its arc position with its "reason"; **Camelot compatibility** lights green/amber/red between adjacent tracks, BPM deltas on connectors; drag-reorder re-fits the arc; swap suggestions ranked by arc+key+mood fit.
- **Honesty:** probabilistic attributes (aux percussion, technique, register) render in lower-contrast/dashed style with "estimated" tooltips; solid ones (BPM, key, vocal/instrumental, common instruments) in full accent.

## 8. Risks / honesty
- **Licenses:** MERT/MuQ/MuQ-MuLan weights are **CC-BY-NC** (fine for personal/local; keep a permissive-only profile — CLAP + AudioSet taggers + Demucs are MIT/Apache — if ever commercial).
- **"Cowbell" on dense mixes** is the weakest link → surface as probabilistic, gate stem-separation behind a toggle.
- **Voice type (SATB) is unsolved from a mix** → register = coarse low/mid/high hint only. **Acoustic** region/dialect classification is also unsolved — but the artist's **region of origin IS available via track identification → MusicBrainz/Discogs metadata** (see §9b). We get region from *who it is*, not from the sound.
- **CLAP cosine is uncalibrated & prompt-sensitive** → per-query calibration + ReCLAP prompts are **mandatory**, not polish.
- **Domain shift:** Essentia gender, VocalSet technique, Whisper LID are trained on cleaner/solo/speech → treat as probabilities, prefer the separated stem.
- **Compute:** ingest taggers are fine; the deep pass (HTDemucs + Whisper + technique) is seconds/track → batch, cache, opt-in.
- **Dependency surface:** wrap each model behind the existing lazy-import `Embedder`/head pattern with install hints; prefer an **ONNX path for Essentia** to avoid a hard TensorFlow dep.
- **LLM caption hallucination** → structured tags are the source of truth; constrain the LLM to only describe provided facts (or template it).

## 8b. Track identification, mix tracklisting & region (metadata path)
Identity unlocks the metadata-only attributes (artist region/origin, year, label, official genre) that audio can't give.

**Three identification tiers (router, cheapest → most powerful):**
1. **In-library match (no network):** embed the query → cosine vs the library's cached embeddings → the track if it's already yours. Already built (`identify_in_library`). Great for "what was that in my own collection".
2. **Global acoustic fingerprint (open):** **Chromaprint** (`fpcalc`) → **AcoustID** API → MusicBrainz recording/artist. Free, but coverage is patchy and needs the `fpcalc` binary + an API key.
3. **Commercial recognition (best coverage, incl. obscure/live):** **AudD** or **ACRCloud** (Shazam-grade). Paid API keys; catches a lot that AcoustID/Shazam-lite miss, and supports continuous/stream recognition.

Once identified → **MusicBrainz** (no key, just a UA + rate-limit) for the artist's `area`/`begin-area` (**region/origin**), plus year/label/official tags; optionally Discogs/Wikidata. This is how "what region is this voice from" actually gets answered.

**Mix / DJ-set tracklisting with timestamps** ("shove a whole set in → every song + when"):
- Slide a window (e.g. 10–20 s, hop 5 s) over the mix; identify each window via the router above (in-library embedding match first — *free and instant for tracks you own* — then fingerprint/commercial for unknowns).
- **Merge consecutive windows** that resolve to the same track into segments → a timestamped tracklist `[{start, end, track_id|external_id, title, artist, confidence}]`; flag overlap/transition regions where two tracks score highly (the actual mix points).
- Robustness notes: DJ mixes are pitch/tempo-shifted and EQ'd, so exact fingerprinting degrades — embedding-similarity matching against the user's own library is the strong path; commercial APIs (ACRCloud has a dedicated broadcast/“humming”+continuous mode) handle the rest. Expose confidence + the transition regions honestly.
- API: `POST /api/identify-mix {path}` → segments; UI: drop a mix → a timeline of identified tracks with timestamps + jump-to-time playback.

**Mobile companion (record-and-identify on the phone):**
- Tauri v2 builds to **iOS/Android** from the same codebase, OR a lightweight **PWA**. The phone records audio → either (a) streams to the desktop engine on the LAN, or (b) runs a slim on-device identify (in-library embedding match needs the embedder on-device; fingerprint/commercial only needs to upload a snippet).
- Use cases the user wants: record a track Shazam can't get (match against *your* library / obscure DBs), record a friend's live mix and get the tracklist, quick "what's this" capture synced back to the desktop library.
- Scope: real but heavier (mobile build, recording permissions, on-device vs server inference). A **record→upload→identify** PWA/companion is the pragmatic v1; full on-device embedding is a later optimization.

## 9. Build phases
1. **Keystone — tags + open-vocab search:** AudioSet tagger (AST/EfficientAT) → `understanding.audioset` + instrument/vocal chips; **open-vocab search** (CLAP mean+chunkmax + calibration + router) → `/api/search` + a search bar. ("songs with cowbells" works.)
2. **Identification + region:** extend identify with the **router** (in-library → AcoustID/Chromaprint → AudD/ACRCloud) and **MusicBrainz** lookup → artist **region/origin**, year, label. ("what region is this voice from" works once the track is named.)
3. **Mix tracklisting:** `/api/identify-mix` — windowed identification over a whole set → **timestamped tracklist** + transition regions; a drop-a-mix → timeline UI.
4. **Rich record:** Essentia voice/gender + mood heads; MERT MLP probes (OpenMIC, MTG-Jamendo); LLM-over-tags caption + canonical tags; per-song **detail view**.
5. **Deep pass:** HTDemucs stems; Whisper language; CREPE register; VocalSet technique — gated toggle.
6. **Set-builder fusion** (attribute + region/language constraints + Camelot/mood tie-breakers) and the **UI overhaul** (home, detail view, constellation, timeline).
7. **Mobile companion:** Tauri-mobile/PWA record-and-identify (record → upload → identify against your library + global DBs; sync back to desktop).
8. **Encoder A/B:** MuQ vs MERT, MuQ-MuLan vs CLAP behind the `Embedder` interface.

## 10. Key sources
- **MERT** — Acoustic Music Understanding via SSL, ICLR 2024, arXiv:2306.00107 · hf.co/m-a-p/MERT-v1-330M
- **MuQ / MuQ-MuLan** — Mel-RVQ SSL, MARBLE SOTA encoder + zero-shot tagging SOTA, 2025, arXiv:2501.01108 · github.com/tencent-ailab/MuQ
- **LAION-CLAP** — contrastive language-audio, ICASSP 2023, arXiv:2211.06687 · github.com/LAION-AI/CLAP
- **EfficientAT** — Transformer-to-CNN KD AudioSet tagging, ICASSP 2023, arXiv:2211.04772 · github.com/fschmid56/EfficientAT
- **AST** — Audio Spectrogram Transformer (AudioSet-527, incl. *Cowbell*), Interspeech 2021, arXiv:2104.01778 · hf.co/MIT/ast-finetuned-audioset-10-10-0.4593
- **HTDemucs / Demucs v4** — hybrid-transformer source separation, ICASSP 2023 · github.com/facebookresearch/demucs
- **OpenMIC-2018** — multi-instrument dataset, ISMIR 2018 · github.com/cosmir/openmic-2018
- **Essentia models** — Discogs-EffNet + voice/gender/mood/arousal-valence heads · essentia.upf.edu/models
- **VocalSet** — singing-technique dataset, ISMIR 2018; technique task in MARBLE (NeurIPS 2023, arXiv:2306.10548)
- **Singing Language ID (deep phonotactic)** — Deezer, ICASSP 2021, arXiv:2105.15014
- **ReCLAP** — descriptive prompts for zero-shot audio, 2024, arXiv:2409.09213
- **Whisper large-v3** — robust ASR / language ID, 2023 · hf.co/openai/whisper-large-v3
- **MARBLE** — music representation benchmark, NeurIPS 2023, arXiv:2306.10548
