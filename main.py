import sys
import os
import logging
from typing import List
from dataclasses import asdict
from src.audio_processing.processor import AudioProcessor
from src.speech_to_text.transcriber import TranscriptSegment, transcribe_with_speakers
from src.gui.main_gui import run_gui

# Configure logging at the module level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Get logger at module level
logger = logging.getLogger(__name__)

# Removed configure_logging() function as config is now at module level

def process_meeting_audio(audio_path: str) -> List[TranscriptSegment]:
    """Process an audio file through the full pipeline.
    
    Args:
        audio_path: Path to the audio file to process
        
    Returns:
        List of transcript segments with speaker information
    """
    try:
        # Initialize audio processor
        processor = AudioProcessor()
        
        # Load and process audio (includes diarization)
        audio, sr = processor.load_audio(audio_path)
        
        # Get speaker segments from diarization
        speaker_segments = processor.get_speaker_segments()
        
        # Transcribe with speaker information
        transcript_segments = transcribe_with_speakers(
            audio_path,
            speaker_segments,
            language="en"  # TODO: Make configurable
        )
        
        logger.debug(f"process_meeting_audio returning {len(transcript_segments)} segments.")
        if transcript_segments:
            # Use asdict if available, otherwise just log the object type/repr
            try:
                from dataclasses import asdict
                logger.debug(f"First segment data: {asdict(transcript_segments[0])}")
            except ImportError:
                 logger.debug(f"First segment object: {transcript_segments[0]}")

        return transcript_segments
        
    except Exception as e:
        logging.error(f"Error processing audio file {audio_path}: {e}")
        raise

def main():
    """Main entry point for the application."""
    # Logging is configured at module level, logger is defined globally
    
    try:
        # Ensure the src directory is in the Python path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(script_dir, 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            
        logger.info("Starting Meeting Transcriber...")
        
        # Launch GUI with processing callback
        run_gui(process_callback=process_meeting_audio)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        logger.info("Meeting Transcriber stopped.")

if __name__ == "__main__":
    main()