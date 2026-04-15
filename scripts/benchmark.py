#!/usr/bin/env python3
"""Benchmark: time and memory profiling for the clotaire pipeline."""

from __future__ import annotations

import sys
import time
import tracemalloc


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file> [--hf-token TOKEN]")
        sys.exit(1)

    audio_path = sys.argv[1]
    hf_token = None
    if "--hf-token" in sys.argv:
        idx = sys.argv.index("--hf-token")
        hf_token = sys.argv[idx + 1]

    print(f"Benchmarking: {audio_path}")
    print("=" * 60)

    tracemalloc.start()
    t_start = time.perf_counter()

    from clotaire import run

    result = run(audio_path, hf_token=hf_token)

    t_end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = t_end - t_start
    duration = result["duration"]
    langs = ", ".join(f"{lang} ({dur}s)" for lang, dur in result["languages"].items())
    n_speakers = len({s["speaker"] for s in result["diarization"]})
    n_words = len(result["words"])

    print(f"Audio duration:  {duration}s")
    print(f"Languages:       {langs}")
    print(f"Speakers:        {n_speakers}")
    print(f"Words:           {n_words}")
    print(f"Elapsed time:    {elapsed:.1f}s")
    print(f"Real-time ratio: {elapsed / duration:.1f}×")
    print(f"Peak memory:     {peak / 1024 / 1024:.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
