# Musik

Musik is the tool I built to deal with my own DJ library. I have around 1500 tracks, mostly electronic, and none of them had real genres on them. "House" tells me nothing when I'm trying to find the right deep, groovy thing for sunset versus a punchy peak-time roller. So I wanted something that listens to the actual audio, sorts it into the genres and subgenres I care about, and writes tags Rekordbox can read. It runs locally on my GPU and nothing gets uploaded anywhere.

The whole idea is that you define genres by example. You are not picking from a fixed list someone else decided on. You give it a handful of tracks that sound like "deep dub techno" to you and it learns that from the audio. Want a new subgenre, drop in a few more examples. No retraining, no waiting.

## What it does

You drag a folder of music in (or hit Browse, or drop files straight onto the window). Musik fingerprints every track with a music model, and from there it can:

- Sort everything into your own Genre/Subgenre folders and write Rekordbox-ready genre tags.
- Classify by example: a centroid or nearest-neighbor match over the embeddings of your reference tracks. When you correct something it gets folded back in and the next pass is sharper.
- Find sound-alikes ("tracks like this one"), cluster an unsorted dump into groups you can name, and plot the whole library as a 2D map.
- Work out BPM, musical key and energy for every track.
- Tag the actual sounds in a track using AudioSet (527 classes, including instruments, vocals, and yes, a literal cowbell class).
- Search by sound in plain words. Type "songs with cowbells" or "female vocals" or "dark melodic techno around 124 bpm" and it ranks your library. Known sounds go through the precise tagger, anything else goes through CLAP text-to-audio matching, and you can drag a threshold to pull every match instead of just the top few.
- Give each track a profile: the instruments it heard, vocal or instrumental with a rough gender read, a mood (it places the track on a valence/arousal map and hands you the closest named moods, things like driving, hypnotic, dark, dreamy), and a one-line description.
- Build DJ sets from a description. Tell it "light groovy house at sunset, start slow, build punchier, then slow down deep and minimal" and it orders a set along that energy and BPM curve. Add constraints like "female vocal, instrumental warmup, with guitar, no ambient" and it filters by what it actually heard, then keeps neighbouring tracks in compatible keys using the Camelot wheel.
- Identify a track by its sound against your own library, and give you a radio queue of what to play next that auto-advances.
- Tracklist a whole mix. Drop a recorded set in and it tells you which of your tracks are in it and at what timestamps.
- Pull the artist's region and origin from MusicBrainz once a track is identified. Region comes from who made it, not from guessing at the audio.

There is also an optional deep pass that splits a track into stems (Demucs), re-listens to the drum stem to catch quiet percussion, and runs the vocal stem through Whisper to guess the sung language. It is slower and off by default.

Anything that touches your files has a dry-run preview and one-click undo. The database is the source of truth. Files only move when you say so.

## On your phone

You can open Musik on your phone and install it like a normal app. There is a Record and Identify screen: tap record, it listens for about ten seconds, and matches what it heard against your library. Point it at the engine running on your desktop over your local network. Good for working out what a track is, or what someone is mixing, as long as it is something in your own crate.

## Setup (Windows, NVIDIA GPU)

One command sets up everything: the Python engine, CUDA PyTorch for your GPU, the music model, the app, and a Musik shortcut on your desktop.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Then double-click the Musik icon. The app opens and the engine starts on its own, there is nothing else to run. The first launch builds the app once (a minute or two), after that it is instant.

You need Python 3.13, Node 18 or newer, and Rust (for the desktop shell) installed first. Your NVIDIA GPU is used automatically through the CUDA build of PyTorch.

### Picking a model

Set it in the top bar.

- mert is the one to use. It is trained on music and it is sharp on fine electronic subgenres. Runs on the GPU.
- baseline needs no download and is rough. Fine for a first look.
- muq is the newest option and currently the strongest on the MARBLE benchmark. Worth A/B testing against mert on your own labels.
- discogs and clap are also there (Discogs-style labels, and defining a genre by text). See engine/README.md.

The bigger models download from Hugging Face the first time you use them. The deep pass (stems and language) needs Demucs and Whisper, which you install once with `pip install demucs openai-whisper`.

## How the classification actually works

There is no fixed classifier that you would have to retrain every time you want a new genre. Instead:

A music foundation model (MERT by default) turns each track into a vector that captures its timbre, rhythm and texture. A genre is just the average of the vectors of the tracks you gave as examples. To classify a track, Musik measures cosine similarity to those averages. Adding a subgenre is dropping in a few examples, and your confirmations become new examples, so it improves as you use it.

For the open-vocabulary side, CLAP puts audio and text in one shared space, so a text query can rank audio directly. AudioSet tagging supplies the precise, named sounds. BPM, key and energy come from ordinary signal processing (numpy and scipy), no heavy models needed.

MusicBrainz is the authoritative metadata layer. Once a track is identified, its real genres, year, label and the artist's region come from MusicBrainz, whose data is CC0 and free to use.

## Layout

- engine/ is the Python side: embeddings, the by-example classifier, clustering, analysis, tagging, search, the set builder, identify, tags and organize, and a local FastAPI server the app talks to. It has a full test suite. See engine/README.md.
- app/ is the desktop UI (Tauri and React). The Rust shell starts the engine for you. See app/README.md.
- docs/ and research/ hold the design notes and the research the build is based on.

## Privacy

It all runs on your machine and your audio never leaves your computer. The only network calls are optional ones: model downloads from Hugging Face the first time, and MusicBrainz or AcoustID lookups when you ask to identify a track.
