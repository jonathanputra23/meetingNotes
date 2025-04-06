import librosa
import soundfile as sf
import io

SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a']

def load_audio(file_path: str, target_sr: int = 16000):
    """
    Loads an audio file, converts it to mono WAV format with the target sample rate.

    Args:
        file_path (str): Path to the input audio file.
        target_sr (int): Target sample rate for the output audio.

    Returns:
        np.ndarray: The audio time series as a NumPy array.
        int: The sample rate of the audio time series.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    # Basic validation (can be expanded)
    if not any(file_path.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
        raise ValueError(f"Unsupported file format. Supported formats: {SUPPORTED_FORMATS}")

    try:
        # Load audio file using librosa. This handles various formats.
        # Librosa automatically resamples to target_sr if specified.
        # It also converts to mono by default.
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        print(f"Loaded audio from {file_path}, sample rate: {sr}, duration: {librosa.get_duration(y=audio, sr=sr):.2f}s")
        return audio, sr
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        raise

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create a dummy WAV file for testing
    import numpy as np
    import os

    dummy_sr = 44100
    dummy_duration = 5 # seconds
    dummy_freq = 440 # Hz
    dummy_audio = 0.5 * np.sin(2 * np.pi * dummy_freq * np.arange(dummy_sr * dummy_duration) / dummy_sr)
    dummy_file = "dummy_audio.wav"

    sf.write(dummy_file, dummy_audio, dummy_sr)
    print(f"Created dummy file: {dummy_file}")

    try:
        loaded_audio, loaded_sr = load_audio(dummy_file)
        print(f"Successfully loaded dummy audio. Shape: {loaded_audio.shape}, Sample Rate: {loaded_sr}")

        # Test with a non-existent file
        # load_audio("non_existent_file.wav")

        # Test with an unsupported format (create dummy)
        dummy_txt_file = "dummy_audio.txt"
        with open(dummy_txt_file, "w") as f:
            f.write("This is not audio.")
        # load_audio(dummy_txt_file)

    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        if os.path.exists(dummy_txt_file):
            os.remove(dummy_txt_file)
        print("Cleaned up dummy files.")