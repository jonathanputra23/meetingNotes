import subprocess
import os
import tempfile
import platform
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging # Import logging
from dataclasses import dataclass

# Get logger for this module
logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

# --- Configuration ---
WHISPER_CPP_EXECUTABLE = "/Users/jonathan/Documents/Github/MeetingNotes/whisper.cpp/whisper-cli"
WHISPER_CPP_MODEL_PATH = "/Users/jonathan/Documents/Github/MeetingNotes/whisper.cpp/models/ggml-base.en.bin"

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
        print("Download models from https://huggingface.co/ggerganov/whisper.cpp/tree/main")
        return False
    return True

import re # Import regex for VTT parsing

def _parse_vtt_timestamp(ts: str) -> float:
    """Converts VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    parts = ts.split(':')
    if len(parts) == 3: # HH:MM:SS.mmm
        h, m, s_ms = parts
        s, ms = s_ms.split('.')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    elif len(parts) == 2: # MM:SS.mmm
        m, s_ms = parts
        s, ms = s_ms.split('.')
        return int(m) * 60 + int(s) + int(ms) / 1000.0
    else:
        raise ValueError(f"Invalid VTT timestamp format: {ts}")

def _parse_timestamped_output(output_file: str) -> List[Dict]:
    """Parse whisper.cpp VTT output file into segments."""
    segments = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"VTT output file not found: {output_file}")
        return []
    except Exception as e:
        logger.error(f"Error reading VTT file {output_file}: {e}")
        return []

    # Simple VTT parser state machine
    state = "find_timestamp" # States: find_timestamp, read_text
    current_segment = {}
    text_buffer = []

    # Regex to match VTT timestamp lines
    timestamp_pattern = re.compile(r"(\d{2}:)?\d{2}:\d{2}\.\d{3}\s+-->\s+(\d{2}:)?\d{2}:\d{2}\.\d{3}")

    for line_num, line in enumerate(lines):
        line = line.strip()

        if state == "find_timestamp":
            match = timestamp_pattern.match(line)
            if match:
                try:
                    start_ts_str = match.group(0).split(' --> ')[0]
                    end_ts_str = match.group(0).split(' --> ')[1]
                    current_segment['start'] = _parse_vtt_timestamp(start_ts_str)
                    current_segment['end'] = _parse_vtt_timestamp(end_ts_str)
                    text_buffer = [] # Reset text buffer for new segment
                    state = "read_text"
                except ValueError as e:
                    logger.warning(f"Error parsing VTT timestamp line {line_num+1}: {line} - {e}")
                    current_segment = {} # Reset segment on error
                except Exception as e:
                    logger.error(f"Unexpected error processing VTT timestamp line {line_num+1}: {line} - {e}")
                    current_segment = {} # Reset segment on error
            # Skip WEBVTT header, empty lines, or sequence numbers before timestamps
            elif line and not line.startswith("WEBVTT") and not line.isdigit():
                 logger.debug(f"Skipping unexpected line while looking for timestamp: {line}")

        elif state == "read_text":
            if line: # Collect text lines
                text_buffer.append(line)
            else: # Blank line signifies end of text for the current segment
                if current_segment and text_buffer:
                    current_segment['text'] = " ".join(text_buffer)
                    # Basic validation
                    if current_segment['start'] >= 0 and current_segment['end'] >= current_segment['start']:
                         segments.append(current_segment)
                    else:
                         logger.warning(f"Skipping invalid parsed segment time: start={current_segment.get('start')}, end={current_segment.get('end')}")
                elif current_segment and not text_buffer:
                     logger.warning(f"Found timestamp but no text for segment ending around {current_segment.get('end')}")

                # Reset for next segment
                current_segment = {}
                text_buffer = []
                state = "find_timestamp"

    # Handle case where file ends with text lines without a trailing blank line
    if state == "read_text" and current_segment and text_buffer:
         current_segment['text'] = " ".join(text_buffer)
         if current_segment['start'] >= 0 and current_segment['end'] >= current_segment['start']:
              segments.append(current_segment)
         else:
              logger.warning(f"Skipping invalid parsed segment time at EOF: start={current_segment.get('start')}, end={current_segment.get('end')}")

    return segments


# --- Ensure the function definition ends correctly ---

# The following lines are just context and should remain unchanged
# def _merge_speaker_segments(...):
# ...

# --- Correcting the end of the _parse_timestamped_output function ---
# The original SEARCH block incorrectly included lines 51-56.
# This diff ensures the function structure is correct after replacement.

# Find the original line 56 and ensure the return is correctly placed after the loop.
# If line 56 was `return segments`, it should be moved outside the loop.
# The replacement block above already includes the correct return statement placement.
# No explicit SEARCH/REPLACE needed here if the above block correctly replaces lines 44-56.

# --- Final check on function structure ---
# Ensure the indentation level of the final `return segments` matches the `for line in f:` loop start.
# The replacement block handles this.
# Removing leftover incorrect code from previous diff attempt.

def _merge_speaker_segments(
    whisper_segments: List[Dict],
    speaker_segments: List[Dict]
) -> List[TranscriptSegment]:
    """Merge whisper segments with speaker diarization results."""
    merged = []
    speaker_idx = 0
    
    for seg in whisper_segments:
        # Find overlapping speaker segments
        current_speaker = None
        while (speaker_idx < len(speaker_segments) and 
               speaker_segments[speaker_idx]['end'] <= seg['start']):
            speaker_idx += 1
            
        if speaker_idx < len(speaker_segments):
            speaker_seg = speaker_segments[speaker_idx]
            if (speaker_seg['start'] <= seg['end'] and 
                speaker_seg['end'] >= seg['start']):
                overlap_start = max(seg['start'], speaker_seg['start'])
                overlap_end = min(seg['end'], speaker_seg['end'])
                overlap_duration = overlap_end - overlap_start
                seg_duration = seg['end'] - seg['start']
                
                if overlap_duration / seg_duration > 0.5:  # Majority overlap
                    current_speaker = speaker_seg['speaker']
        
        merged.append(TranscriptSegment(
            start=seg['start'],
            end=seg['end'],
            text=seg['text'],
            speaker=current_speaker
        ))
    
    return merged

def transcribe_with_speakers(
    audio_file_path: str,
    speaker_segments: List[Dict],
    language: str = "en"
) -> List[TranscriptSegment]:
    """
    Transcribe audio with speaker labels.
    
    Args:
        audio_file_path: Path to audio file
        speaker_segments: Speaker segments from diarization
        language: Language code
        
    Returns:
        List of transcript segments with speaker labels
    """
    # 1. Run whisper with timestamped output
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_output:
        output_file = temp_output.name
        output_base = os.path.splitext(output_file)[0]

    command = [
        WHISPER_CPP_EXECUTABLE,
        "-m", WHISPER_CPP_MODEL_PATH,
        "-l", language,
        "-f", audio_file_path,
        "-ovtt", # Changed to VTT output
        "-of", output_base,
        "-t", "8",  # Use 8 threads
    ]

    try:
        subprocess.run(command, check=True)
        timestamped_file = output_base + ".vtt" # Expect .vtt file
        if not os.path.exists(timestamped_file):
            logger.error(f"Whisper output file not found after execution: {timestamped_file}")
            return [] # Return empty if output file doesn't exist

        logger.debug(f"Parsing whisper output from: {timestamped_file}")
        whisper_segments = _parse_timestamped_output(timestamped_file)
        logger.debug(f"Parsed {len(whisper_segments)} segments from whisper output.")

        if not whisper_segments:
             logger.warning("Whisper parsing returned no segments.")
             # If whisper produced output but parsing failed, this indicates a parsing logic error
             # or unexpected whisper output format.
             return [] # Return empty if parsing failed

        logger.debug(f"Merging {len(whisper_segments)} whisper segments with {len(speaker_segments)} speaker segments.")
        merged_segments = _merge_speaker_segments(whisper_segments, speaker_segments)
        logger.debug(f"Returning {len(merged_segments)} merged segments.")
        return merged_segments
    except subprocess.CalledProcessError as e:
        # Log detailed error if subprocess fails
        logger.error(f"Whisper subprocess failed with exit code {e.returncode}")
        logger.error(f"Command: {' '.join(e.cmd)}")
        # Capture stderr and stdout if available (might be None)
        stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "N/A"
        stdout_output = e.stdout.decode('utf-8', errors='ignore') if e.stdout else "N/A"
        logger.error(f"Stderr: {stderr_output}")
        logger.error(f"Stdout: {stdout_output}")
        raise # Re-raise the exception to be caught by the GUI
    except Exception as e:
        logger.error(f"Error during transcription or merging: {e}", exc_info=True)
        raise # Re-raise for the GUI
    finally:
        # Safer cleanup: Use separate try/except for each file removal
        try:
            if 'output_file' in locals() and os.path.exists(output_file):
                os.remove(output_file)
                logger.debug(f"Removed temp file: {output_file}")
        except OSError as e:
            logger.warning(f"Could not remove temp file {output_file}: {e}")
        
        # timestamped_file might not be defined if subprocess failed early
        timestamped_file_to_remove = output_base + ".vtt" # Clean up .vtt file
        try:
            # Check existence again, as it might have been removed or never created
            if 'output_base' in locals() and os.path.exists(timestamped_file_to_remove):
                os.remove(timestamped_file_to_remove)
                logger.debug(f"Removed whisper output file: {timestamped_file_to_remove}")
        except OSError as e:
             logger.warning(f"Could not remove whisper output file {timestamped_file_to_remove}: {e}")

# Maintain backward compatibility
def transcribe_audio(audio_file_path: str, language: str = "en") -> str:
    """Original transcription function for backward compatibility."""
    segments = transcribe_with_speakers(audio_file_path, [], language)
    return "\n".join([seg.text for seg in segments])