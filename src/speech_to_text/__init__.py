# This file makes the speech_to_text directory a Python package.
from .transcriber import transcribe_audio

__all__ = ['transcribe_audio']