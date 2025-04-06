import sys
import os

# Ensure the src directory is in the Python path
# This allows the script to find the modules within src
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir) # Prepend src dir to path

try:
    # Import the main function from the gui module
    from gui import main as run_gui
except ImportError as e:
    print(f"Error importing GUI module: {e}")
    print("Please ensure you are running this script from the 'meeting_transcriber' directory", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    print("Starting Meeting Transcriber GUI...")
    run_gui()
    print("Meeting Transcriber GUI closed.")