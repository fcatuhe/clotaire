#!/usr/bin/env python3
"""Pre-download all models to local cache.

Run this once on a new machine to avoid waiting during first transcription.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Set HF_TOKEN env var before running this script.")
        print("  export HF_TOKEN=hf_xxxxxxxx")
        sys.exit(1)

    print("Downloading whisper large-v3...")
    import whisper
    whisper.load_model("large-v3")
    print("  ✓ whisper large-v3 cached")

    print("Downloading pyannote segmentation...")
    from pyannote.audio import Model
    Model.from_pretrained("pyannote/segmentation")
    print("  ✓ pyannote segmentation cached")

    print("Downloading pyannote diarization...")
    from pyannote.audio import Pipeline
    Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
    print("  ✓ pyannote diarization cached")

    print("Downloading wav2vec2 alignment models (fr, en)...")
    from whisperx.alignment import load_align_model
    for lang in ["fr", "en"]:
        load_align_model(lang, "cpu")
        print(f"  ✓ wav2vec2 alignment ({lang}) cached")

    print("\nAll models downloaded successfully!")


if __name__ == "__main__":
    main()
