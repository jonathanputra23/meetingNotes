import librosa
import soundfile as sf
import io
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path
from ..speaker_diarization import SpeakerDiarizer
from typing import Optional

SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a']

class AudioProcessor:
    def __init__(self):
        """Initialize audio processor with diarization capability."""
        self.diarizer = SpeakerDiarizer()
        self._audio = None
        self._sample_rate = None
        self._speaker_segments = None

    def load_audio(self, file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Loads and processes an audio file, including speaker diarization.
        
        Args:
            file_path: Path to the input audio file
            target_sr: Target sample rate (default 16000)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Validate file format
        if not any(str(file_path).lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
            raise ValueError(f"Unsupported file format. Supported formats: {SUPPORTED_FORMATS}")

        try:
            # Load and process audio
            self._audio, self._sample_rate = librosa.load(
                file_path, 
                sr=target_sr, 
                mono=True
            )
            
            # Perform speaker diarization and store segments
            try:
                segments = self.diarizer.process_audio(file_path)
                print(f"Diarization returned {len(segments)} segments")  # Debug
                print(f"First segment: {segments[0] if segments else 'None'}")  # Debug
                self._speaker_segments = segments
            except Exception as diarize_err:
                raise RuntimeError(f"Diarization failed: {str(diarize_err)}") from diarize_err
            
            return self._audio, self._sample_rate
            
        except Exception as e:
            print(f"Error processing audio file {file_path}: {e}")
            raise

    def get_speaker_segments(self) -> List[Dict]:
        """Get the speaker segments identified during processing.
        
        Returns:
            List of speaker segments with start/end times and speaker labels
        """
        if self._speaker_segments is None:
            raise RuntimeError("No audio processed yet. Call load_audio() first.")
        return self._speaker_segments

    def get_audio_segment(self, start_time: float, end_time: float) -> np.ndarray:
        """Extract a segment of the loaded audio.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio segment as numpy array
        """
        if self._audio is None:
            raise RuntimeError("No audio loaded. Call load_audio() first.")
            
        start_sample = int(start_time * self._sample_rate)
        end_sample = int(end_time * self._sample_rate)
        return self._audio[start_sample:end_sample]

# Maintain backward compatibility with standalone function
def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Standalone function for backward compatibility."""
    processor = AudioProcessor()
    return processor.load_audio(file_path, target_sr)