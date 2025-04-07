import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
from typing import Callable, List, Optional
from dataclasses import asdict
# Use relative imports within the package structure
from ..speech_to_text.transcriber import TranscriptSegment
from ..speaker_diarization.diarizer import SpeakerDiarizer

class MeetingTranscriberApp:
    """Main GUI application for the Meeting Transcriber."""
    
    def __init__(self, root, process_callback: Optional[Callable] = None):
        """Initialize the GUI application.
        
        Args:
            root: Tk root window
            process_callback: Function to call for audio processing
        """
        self.root = root
        self.process_callback = process_callback
        self.root.title("Meeting Transcriber")
        self.root.geometry("800x600")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TButton', padding=6)
        self.style.configure('TLabel', padding=6)
        
        # Create main widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        # File selection frame
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.pack(fill=tk.X)
        
        ttk.Label(file_frame, text="Audio File:").pack(side=tk.LEFT)
        self.file_entry = ttk.Entry(file_frame, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        
        browse_btn = ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.browse_file
        )
        browse_btn.pack(side=tk.LEFT)
        
        # Process button
        process_btn = ttk.Button(
            self.root,
            text="Transcribe Meeting",
            command=self.start_transcription
        )
        process_btn.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Results display
        self.result_text = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=('Consolas', 10)
        )
        self.result_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        
    def browse_file(self):
        """Open file dialog to select audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Meeting Recording",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            
    def start_transcription(self):
        """Start transcription process in a separate thread."""
        file_path = self.file_entry.get()
        if not file_path:
            self.show_error("Please select an audio file first")
            return
            
        if not self.process_callback:
            self.show_error("Processing function not available")
            return
            
        self.update_status("Processing...")
        self.result_text.delete(1.0, tk.END)
        
        # Run processing in background thread
        thread = threading.Thread(
            target=self.transcription_thread,
            args=(file_path,),
            daemon=True
        )
        thread.start()
        
    def transcription_thread(self, file_path):
        """Thread function for processing audio."""
        try:
            segments = self.process_callback(file_path)
            self.root.after(0, self.show_result, segments)
            self.root.after(0, self.update_status, "Done")
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
            
    def update_status(self, message):
        """Update status bar message."""
        self.status_var.set(message)
        
    def show_result(self, segments: List[TranscriptSegment]):
        """Display transcription results with speaker labels."""
        self.result_text.delete(1.0, tk.END)
        
        for segment in segments:
            speaker = segment.speaker or "Unknown"
            time_str = f"[{segment.start:.1f}-{segment.end:.1f}s]"
            self.result_text.insert(tk.END, f"{time_str} {speaker}:\n")
            self.result_text.insert(tk.END, f"{segment.text}\n\n")
            
    def show_error(self, error_message):
        """Display error message."""
        self.status_var.set(f"Error: {error_message}")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Error: {error_message}")

def run_gui(process_callback: Optional[Callable] = None):
    """Run the GUI application.
    
    Args:
        process_callback: Function to call for audio processing
    """
    root = tk.Tk()
    app = MeetingTranscriberApp(root, process_callback)
    root.mainloop()