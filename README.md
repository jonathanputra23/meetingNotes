# Meeting Transcriber Project

![Project Status](https://img.shields.io/badge/Status-MVP%20Complete-green)

## Current Implementation Status

```mermaid
pie
    title Feature Completion
    "Completed" : 45
    "In Progress" : 15
    "Pending" : 40
```

### ✅ Implemented Features
- Audio processing (WAV/MP3/M4A support)
- Whisper.cpp English transcription
- Basic PySimpleGUI interface
- File selection and processing workflow

### 🚧 In Development
- Speaker diarization (Pyannote-audio integration)
- Indonesian language support

### 📅 Planned Features
- AI summarization (Ollama integration)
- Action item extraction
- Enhanced output formats

## System Architecture

```mermaid
graph TD
    A[Audio Input] --> B(Audio Processing)
    B --> C[Speech-to-Text]
    C --> D[Speaker Diarization]
    D --> E[Text Analysis]
    E --> F[GUI Display]
    E --> G[File Outputs]
```

## Getting Started

1. Install requirements:
```bash
pip install -r meeting_transcriber/requirements.txt
```

2. Run the application:
```bash
python meeting_transcriber/main.py
```

## Development Roadmap

```mermaid
gantt
    title Development Timeline
    dateFormat  YYYY-MM-DD
    section MVP
    Audio Processing :done, 2025-03-17, 7d
    Transcription :done, 2025-03-24, 7d
    Basic GUI :done, 2025-03-31, 7d

    section Phase 2
    Speaker ID :active, 2025-04-07, 14d
    Indonesian :2025-04-21, 7d

    section Phase 3
    Summarization :2025-04-28, 14d
    Action Items :2025-05-12, 7d
```

## Contributing
Pull requests welcome. Please follow the existing code style and add tests for new features.