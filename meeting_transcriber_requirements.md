# Meeting Transcriber Requirements Document

## 1. Functional Requirements

### 1.1 Core Features
- **Audio Input Processing**
  - Accept common audio formats (WAV, MP3, M4A)
  - Support meeting recordings 30min-2hr duration
  - Handle multiple speakers (2-10)

- **Speaker Diarization**
  - Automatic speaker identification without training
  - Speaker labeling (Speaker 1, Speaker 2, etc.)
  - Minimum 85% accuracy in speaker separation

- **Multilingual Transcription**
  - English and Indonesian support
  - Automatic language detection
  - Timestamped transcripts

- **AI Summarization**
  - Key discussion point extraction
  - Decision tracking
  - Context-aware summarization

- **Action Item Extraction**
  - Task identification ("we should", "let's", "need to")
  - Responsible party assignment
  - Deadline detection (when possible)

### 1.2 Output Formats
- Raw timestamped transcript (.txt)
- Formatted meeting notes (.md)
- Action item list (.csv)
- Summary document (.md)

## 2. Non-Functional Requirements

### 2.1 Performance
- Process 1hr audio in <15min on mid-range CPU
- Memory usage <4GB
- No internet connection required

### 2.2 Compatibility
- macOS, Windows, Linux support
- Basic GUI interface in MVP
- Progressive enhancement in later versions

## 3. Technical Specifications

### 3.3 GUI Framework
- PySimpleGUI for initial implementation
- Responsive design for different screen sizes
- Dark/light mode support

### 3.1 Architecture
- Python 3.10+
- Modular design with clear interfaces between:
  - Audio processing
  - Speech-to-text
  - Diarization
  - Analysis

### 3.2 Key Dependencies
- Whisper.cpp with ggml-base.en.bin model:
  - English language transcription
  - Base model quality (WER ~5-10%)
  - Optimized for CPU inference
  - ~1GB model size
- Pyannote-audio (speaker diarization)
- Ollama (local LLM for summarization)
- Librosa (audio processing)

## 4. Development Milestones

1. **MVP (4 weeks)**
   - Basic audio processing pipeline
   - English transcription
   - Simple text output
   - Basic GUI interface (file selection, progress, results view)

2. **Phase 2 (2 weeks)**
   - Speaker diarization
   - Indonesian support

3. **Phase 3 (2 weeks)**
   - Ollama integration
   - Summary generation

4. **Phase 4 (1 week)**
   - Action item extraction
   - Multiple output formats