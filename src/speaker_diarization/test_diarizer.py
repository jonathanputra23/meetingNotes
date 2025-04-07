import unittest
import os
import numpy as np
from pathlib import Path
from .diarizer import SpeakerDiarizer
from sklearn.cluster import AgglomerativeClustering

class TestSpeakerDiarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diarizer = SpeakerDiarizer()
        base_dir = Path(__file__).parent.parent.parent.parent
        cls.test_audio = str(base_dir / 'whisper.cpp' / 'samples' / 'jfk.mp3')
        if not os.path.exists(cls.test_audio):
            raise unittest.SkipTest(f"Test audio file not found at {cls.test_audio}")

    def test_diarization(self):
        """Test basic diarization functionality"""
        segments = self.diarizer.process_audio(self.test_audio)
        
        # Should return at least one segment
        self.assertGreater(len(segments), 0)
        
        # Each segment should have required fields
        for seg in segments:
            self.assertIn('start', seg)
            self.assertIn('end', seg)
            self.assertIn('speaker', seg)
            self.assertLess(seg['start'], seg['end'])
            
    def test_empty_audio(self):
        """Test handling of empty/invalid audio"""
        # Expect FileNotFoundError when the audio file doesn't exist
        with self.assertRaises(FileNotFoundError):
            self.diarizer.process_audio("nonexistent.mp3")
            
    def test_embedding_failure(self):
        """Test error handling for embedding failures"""
        # Mock embed_utterance to raise an error
        original_embed = self.diarizer.encoder.embed_utterance
        def mock_embed(*args, **kwargs):
            raise ValueError("Mock embedding failure")
            
        self.diarizer.encoder.embed_utterance = mock_embed
        try:
            with self.assertRaises(ValueError):
                self.diarizer.process_audio(self.test_audio)
        finally:
            self.diarizer.encoder.embed_utterance = original_embed
        
    def test_clustering_fallback(self):
        """Test that clustering falls back to Euclidean when cosine fails"""
        # Mock embed_utterance to return numpy array of timestamps
        original_embed = self.diarizer.encoder.embed_utterance
        def mock_embed(*args, **kwargs):
            # Return format: (wav_splits, cont_embeds) - Assuming tuple return is (timestamps, embeddings)
            wav = np.random.rand(16000)  # 1 second of random audio
            cont_embeds = np.random.rand(10, 256)  # 10 random embeddings as numpy array
            # Return timestamps as pairs (start, end) as numpy array
            timestamps = np.linspace(0, 1, 11) # Generate 11 points for 10 segments
            wav_splits = np.array(list(zip(timestamps[:-1], timestamps[1:]))) # Create pairs as numpy array
            return (wav_splits, cont_embeds) # Return in swapped order (timestamps, embeddings)
            
        self.diarizer.encoder.embed_utterance = mock_embed
        try:
            segments = self.diarizer.process_audio(self.test_audio)
            self.assertGreater(len(segments), 0)
        finally:
            self.diarizer.encoder.embed_utterance = original_embed

    def test_speaker_naming(self):
        """Test speaker name assignment"""
        segments = self.diarizer.process_audio(self.test_audio)
        name_map = {'SPEAKER_00': 'JFK'}
        named_segments = self.diarizer.assign_speaker_names(segments, name_map)
        
        # Should have at least one named segment
        self.assertGreater(len(named_segments), 0)
        
        # Check naming was applied
        for seg in named_segments:
            if seg['speaker'] in name_map:
                self.assertEqual(seg['speaker'], name_map['SPEAKER_00'])

if __name__ == '__main__':
    unittest.main()