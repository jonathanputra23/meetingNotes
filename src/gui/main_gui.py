import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import sys
import traceback

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import modules after updating path
try:
    from audio_processing import load_audio, SUPPORTED_FORMATS
    from speech_to_text import transcribe_audio
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"sys.path: {sys.path}")
    messagebox.showerror("Import Error", 
        f"Failed to import necessary modules: {e}\nPlease ensure the project structure is correct and dependencies are installed.")
    sys.exit(1)

class MeetingTranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meeting Transcriber MVP")
        self.active_thread = None
        
        # Main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        ttk.Label(self.main_frame, text="Select Audio File (.wav, .mp3, .m4a):").pack(anchor=tk.W)
        
        file_frame = ttk.Frame(self.main_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path, state='readonly')
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        
        # Transcribe button
        self.transcribe_btn = ttk.Button(self.main_frame, text="Transcribe", 
                                      command=self.start_transcription, state=tk.DISABLED)
        self.transcribe_btn.pack(pady=5)
        
        # Separator
        ttk.Separator(self.main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Status/Results
        ttk.Label(self.main_frame, text="Status / Results:").pack(anchor=tk.W)
        
        self.output_text = tk.Text(self.main_frame, height=15, width=80, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.main_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.main_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
    def browse_file(self):
        filetypes = [("Audio Files", "*.wav *.mp3 *.m4a"), ("All Files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.file_path.set(filename)
            if os.path.exists(filename):
                self.transcribe_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Selected: {os.path.basename(filename)}")
            else:
                self.transcribe_btn.config(state=tk.DISABLED)
                self.status_var.set("Please select a valid audio file.")
    
    def start_transcription(self):
        if self.active_thread is not None:
            messagebox.showinfo("Busy", "Transcription is already in progress.")
            return
            
        file_path = self.file_path.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("File Error", "Please select a valid audio file first.")
            return
            
        self.transcribe_btn.config(state=tk.DISABLED)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Starting transcription process...\n")
        self.status_var.set("Processing...")
        
        # Start transcription thread
        self.active_thread = threading.Thread(
            target=self.transcription_thread,
            args=(file_path,),
            daemon=True
        )
        self.active_thread.start()
        
        # Check thread status periodically
        self.root.after(100, self.check_thread)
    
    def transcription_thread(self, file_path):
        try:
            self.update_status("Starting audio processing...")
            
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            if not any(file_path.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
                raise ValueError(f"Unsupported file format. Supported: {SUPPORTED_FORMATS}")
            
            self.update_status(f"Audio file validated: {os.path.basename(file_path)}")
            self.update_status("Starting transcription (this may take time)...")
            
            # Transcribe
            transcript = transcribe_audio(file_path, language="en")
            
            # Show result
            self.show_result(transcript)
            self.update_status("Transcription finished successfully.")
            
        except Exception as e:
            error_message = f"Error during transcription:\n{traceback.format_exc()}"
            self.show_error(error_message)
            self.update_status("Transcription failed. Check output for details.")
            
        finally:
            self.active_thread = None
            self.root.after(0, lambda: self.transcribe_btn.config(state=tk.NORMAL))
    
    def check_thread(self):
        if self.active_thread and self.active_thread.is_alive():
            self.root.after(100, self.check_thread)
    
    def update_status(self, message):
        self.root.after(0, lambda: self.output_text.insert(tk.END, f"{message}\n"))
        self.root.after(0, lambda: self.status_var.set(message))
        self.root.after(0, lambda: self.output_text.see(tk.END))
    
    def show_result(self, transcript):
        self.root.after(0, lambda: self.output_text.insert(tk.END, 
            f"\n--- Transcription Result ---\n{transcript}\n----------------------------\n"))
        self.root.after(0, lambda: self.output_text.see(tk.END))
    
    def show_error(self, error_message):
        self.root.after(0, lambda: self.output_text.insert(tk.END, 
            f"\n--- ERROR ---\n{error_message}\n-------------\n"))
        self.root.after(0, lambda: self.output_text.see(tk.END))

def main():
    root = tk.Tk()
    root.geometry("800x600")
    app = MeetingTranscriberApp(root)
    root.mainloop()

if __name__ == "__main__":
    # Basic check for whisper.cpp dependencies before launching GUI
    try:
        from speech_to_text.transcriber import _check_executable, _check_model, WHISPER_CPP_EXECUTABLE, WHISPER_CPP_MODEL_PATH
        # Check if executable and model seem accessible
        if not _check_executable(WHISPER_CPP_EXECUTABLE) and not _check_executable(os.path.join(src_dir, "..", "whisper.cpp", WHISPER_CPP_EXECUTABLE)):
            print(f"Warning: Whisper.cpp executable '{WHISPER_CPP_EXECUTABLE}' not found in PATH or expected relative location.")
            messagebox.showwarning("Dependency Check", 
                f"Whisper.cpp executable ('{WHISPER_CPP_EXECUTABLE}') not found.\nPlease ensure whisper.cpp is compiled and accessible.")
        elif not _check_model(WHISPER_CPP_MODEL_PATH) and not _check_model(os.path.join(src_dir, "..", WHISPER_CPP_MODEL_PATH)):
            print(f"Warning: Whisper.cpp model '{WHISPER_CPP_MODEL_PATH}' not found.")
            messagebox.showwarning("Dependency Check", 
                f"Whisper.cpp model ('{WHISPER_CPP_MODEL_PATH}') not found.\nPlease download a model (e.g., ggml-base.en.bin) and place it correctly.")
        else:
            print("Whisper.cpp executable and model appear to be present.")
            main() # Run the GUI
    except ImportError:
        # Error already handled during initial imports
        pass
    except Exception as e:
        messagebox.showerror("Startup Error", f"An unexpected error occurred before starting the GUI:\n{e}")