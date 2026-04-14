"""Pipeline step functions."""

from kloter.steps.convert import load_audio
from kloter.steps.vad import detect_speech
from kloter.steps.whisper import transcribe_segments, attach_language_to_words
from kloter.steps.diarize import diarize
from kloter.steps.align import align_words
from kloter.steps.match import match_speakers
from kloter.steps.format import format_output, write_files, to_markdown

__all__ = [
    "load_audio",
    "detect_speech",
    "transcribe_segments",
    "attach_language_to_words",
    "diarize",
    "align_words",
    "match_speakers",
    "format_output",
    "write_files",
    "to_markdown",
]
