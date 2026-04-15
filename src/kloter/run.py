"""Orchestrator — runs steps sequentially, saves step files and artifacts."""

from __future__ import annotations

import time
from pathlib import Path

from kloter.steps_io import StepWriter
from kloter.step_01_convert import load_audio, probe_audio, save_wav, build_step as build_step_01
from kloter.step_02_vad import detect_speech, build_step as build_step_02


def run(audio_path: Path) -> None:
    """Run steps 01–02 and save numbered step files."""
    writer = StepWriter(audio_path)

    # ── Step 01: Convert ──
    print("Step 01: convert …", flush=True)

    probe = probe_audio(audio_path)
    audio = load_audio(audio_path)

    # Save converted WAV as step artifact (whisper-cli reads it directly)
    wav_path = writer.artifact_path(1, "convert", ".wav")
    save_wav(audio, wav_path)

    step_01 = build_step_01(audio_path, audio, probe, wav_path)
    writer.save(1, "convert", step_01)

    # ── Step 02: VAD ──
    print("Step 02: vad …", flush=True)

    t0 = time.perf_counter()
    raw_segs, padded_segs, model_info = detect_speech(audio)
    elapsed_02 = time.perf_counter() - t0

    step_02 = build_step_02(raw_segs, padded_segs, model_info, audio)
    step_02["wall_time_s"] = round(elapsed_02, 2)
    writer.save(2, "vad", step_02)

    print("Done. Steps saved to:", writer._steps_dir, file=__import__("sys").stderr)
