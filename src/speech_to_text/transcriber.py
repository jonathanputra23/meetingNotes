import subprocess
import os
import tempfile
import platform

# --- Configuration ---
# TODO: Make these configurable, perhaps via a config file or GUI settings
WHISPER_CPP_EXECUTABLE = "/Users/jonathan/Documents/Github/MeetingNotes/whisper.cpp/whisper-cli"  # Absolute path to whisper-cli
WHISPER_CPP_MODEL_PATH = "/Users/jonathan/Documents/Github/MeetingNotes/whisper.cpp/models/ggml-base.en.bin" # Absolute path to model

# --- Helper Functions ---

def _check_executable(executable_path: str) -> bool:
    """Checks if the whisper.cpp executable exists and is executable."""
    if not os.path.exists(executable_path):
        print(f"Error: Whisper.cpp executable not found at '{executable_path}'")
        return False
    if not os.access(executable_path, os.X_OK):
        print(f"Error: Whisper.cpp executable at '{executable_path}' is not executable.")
        return False
    return True

def _check_model(model_path: str) -> bool:
    """Checks if the whisper.cpp model file exists."""
    if not os.path.exists(model_path):
        print(f"Error: Whisper.cpp model not found at '{model_path}'")
        # Provide guidance on obtaining models
        print("Download models from https://huggingface.co/ggerganov/whisper.cpp/tree/main")
        return False
    return True

# --- Core Transcription Function ---

def transcribe_audio(audio_file_path: str, language: str = "en") -> str:
    """
    Transcribes the given audio file using the whisper.cpp executable.

    Args:
        audio_file_path (str): Path to the input audio file (WAV format recommended).
        language (str): Language code for transcription (e.g., 'en', 'id').

    Returns:
        str: The transcribed text.

    Raises:
        FileNotFoundError: If the audio file, executable, or model is not found.
        RuntimeError: If the whisper.cpp process fails.
    """
    print(f"Starting transcription for: {audio_file_path}")

    # 1. Validate paths
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    if not _check_executable(WHISPER_CPP_EXECUTABLE):
         # Attempt to find it relative to the script if not in PATH
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, "..", "..", "whisper.cpp", WHISPER_CPP_EXECUTABLE)
        if platform.system() == "Windows":
             potential_path += ".exe" # Add .exe for Windows
        if _check_executable(potential_path):
             executable_path = potential_path
             print(f"Using whisper.cpp executable found at: {executable_path}")
        else:
             raise FileNotFoundError(f"Whisper.cpp executable '{WHISPER_CPP_EXECUTABLE}' not found in PATH or expected relative location.")
    else:
        executable_path = WHISPER_CPP_EXECUTABLE # Found in PATH

    model_path = WHISPER_CPP_MODEL_PATH
    if not _check_model(model_path):
        # Attempt to find it relative to the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_model_path = os.path.join(script_dir, "..", "..", model_path)
        if _check_model(potential_model_path):
            model_path = potential_model_path
            print(f"Using model found at: {model_path}")
        else:
            raise FileNotFoundError(f"Whisper.cpp model not found at '{WHISPER_CPP_MODEL_PATH}' or expected relative location.")


    # 2. Prepare whisper.cpp command
    # Using a temporary file for output to avoid issues with stdout buffering/parsing
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_output:
        output_file_path = temp_output.name
        # We just need the base name without the extension for whisper.cpp's -of flag
        output_file_base = os.path.splitext(output_file_path)[0]

    command = [
        executable_path,
        "-m", model_path,
        "-l", language,
        "-f", audio_file_path,
        "-otxt", # Output format: plain text
        "-of", output_file_base # Output file name base (whisper.cpp adds .txt)
    ]
    print(f"Executing command: {' '.join(command)}")

    # 3. Run whisper.cpp process
    try:
        process = subprocess.run(
            command,
            check=True,        # Raise exception on non-zero exit code
            capture_output=True, # Capture stdout/stderr
            text=True,         # Decode stdout/stderr as text
            encoding='utf-8'   # Specify encoding
        )
        print("whisper.cpp stdout:")
        print(process.stdout)
        if process.stderr:
            print("whisper.cpp stderr:")
            print(process.stderr)

    except FileNotFoundError:
        print(f"Error: Failed to execute '{executable_path}'. Is whisper.cpp compiled and in your PATH or the expected location?")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error: whisper.cpp process failed with exit code {e.returncode}")
        print("Command:", e.cmd)
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        # Clean up the temp file path variable if process failed before creating it
        if 'output_file_path' in locals() and os.path.exists(output_file_path):
             os.remove(output_file_path)
        raise RuntimeError(f"whisper.cpp execution failed: {e.stderr}") from e
    except Exception as e:
        print(f"An unexpected error occurred during whisper.cpp execution: {e}")
        if 'output_file_path' in locals() and os.path.exists(output_file_path):
             os.remove(output_file_path)
        raise

    # 4. Read transcription from the output file
    expected_output_file = output_file_base + ".txt"
    try:
        if not os.path.exists(expected_output_file):
             raise FileNotFoundError(f"Whisper.cpp output file not found: {expected_output_file}")

        with open(expected_output_file, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()
        print(f"Transcription successful. Output length: {len(transcription)} chars.")
        return transcription
    except Exception as e:
        print(f"Error reading transcription output file {expected_output_file}: {e}")
        raise
    finally:
        # 5. Clean up the temporary output file
        if os.path.exists(expected_output_file):
            os.remove(expected_output_file)
            print(f"Cleaned up temporary file: {expected_output_file}")


# Example usage (optional, for testing with a dummy audio file)
if __name__ == '__main__':
    import numpy as np
    import soundfile as sf

    # Create a dummy WAV file for testing
    dummy_sr = 16000 # Whisper expects 16kHz mono
    dummy_duration = 5 # seconds
    dummy_freq = 440 # Hz
    dummy_audio = 0.5 * np.sin(2 * np.pi * dummy_freq * np.arange(dummy_sr * dummy_duration) / dummy_sr)
    dummy_file = "dummy_speech_test.wav"

    sf.write(dummy_file, dummy_audio, dummy_sr)
    print(f"Created dummy audio file: {dummy_file}")

    # --- !!! IMPORTANT !!! ---
    # This test requires:
    # 1. whisper.cpp compiled as 'main' (or adjust WHISPER_CPP_EXECUTABLE)
    # 2. A whisper model (e.g., ggml-base.en.bin) placed at WHISPER_CPP_MODEL_PATH
    #    (e.g., ./models/ggml-base.en.bin relative to where you run this script)
    # Create the models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Created 'models' directory. Please place a ggml model (e.g., ggml-base.en.bin) inside.")

    print("\n--- Running Transcription Test ---")
    print(f"Executable: {WHISPER_CPP_EXECUTABLE}")
    print(f"Model:      {WHISPER_CPP_MODEL_PATH}")
    print(f"Audio File: {dummy_file}")
    print("---------------------------------")


    if os.path.exists(WHISPER_CPP_MODEL_PATH) and os.path.exists(dummy_file):
         # Basic check if executable might exist (won't work perfectly on Windows without .exe)
         import shutil
         executable_exists = shutil.which(WHISPER_CPP_EXECUTABLE) is not None
         if not executable_exists:
              # Check relative path too
              script_dir = os.path.dirname(os.path.abspath(__file__))
              potential_path = os.path.join(script_dir, "..", "..", "whisper.cpp", WHISPER_CPP_EXECUTABLE)
              if platform.system() == "Windows":
                   potential_path += ".exe"
              executable_exists = os.path.exists(potential_path) and os.access(potential_path, os.X_OK)


         if executable_exists:
            try:
                transcript = transcribe_audio(dummy_file, language="en")
                print("\n--- Transcription Result ---")
                # Whisper.cpp might output silence for a pure sine wave, which is expected.
                print(f"Transcript: '{transcript}'")
                print("----------------------------")
            except (FileNotFoundError, RuntimeError, Exception) as e:
                print(f"\n--- Transcription Test Failed ---")
                print(f"An error occurred: {e}")
                print("Please ensure whisper.cpp is compiled ('main') and the model ('models/ggml-base.en.bin') exists.")
                print("---------------------------------")
         else:
              print("\n--- Skipping Transcription Test ---")
              print(f"Whisper.cpp executable '{WHISPER_CPP_EXECUTABLE}' not found in PATH or expected relative location.")
              print("Please compile whisper.cpp and ensure 'main' is accessible.")
              print("---------------------------------")

    else:
        print("\n--- Skipping Transcription Test ---")
        if not os.path.exists(WHISPER_CPP_MODEL_PATH):
            print(f"Model file not found: {WHISPER_CPP_MODEL_PATH}")
        if not os.path.exists(dummy_file):
             print(f"Dummy audio file not found: {dummy_file}") # Should not happen
        print("---------------------------------")


    # Clean up dummy audio file
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
        print(f"Cleaned up dummy audio file: {dummy_file}")