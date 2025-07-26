import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from preprocessing.face_detection import FaceDetector
from preprocessing.lip_extraction import LipExtractor
from preprocessing.video_utils import VideoProcessor
from feature_extraction.visual_features import VisualFeatureExtractor
from feature_extraction.temporal_features import TemporalEncoder
from synthesis.speech_synthesis import SpeechSynthesizer
from synthesis.vocoder import Vocoder
from utils.logger import setup_logger
from utils.config import load_config

logger = setup_logger(__name__)

class LipSyncAI:
    """Main class for the LipSync AI pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.lip_extractor = LipExtractor(
            target_size=self.config["preprocessing"]["lip_region_size"]
        )
        
        # Models
        self.visual_feature_extractor = VisualFeatureExtractor(
            feature_dim=self.config["feature_extraction"]["visual_feature_dim"]
        ).to(self.device)
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.config["feature_extraction"]["visual_feature_dim"],
            hidden_dim=self.config["feature_extraction"]["temporal_feature_dim"]
        ).to(self.device)
        
        self.speech_synthesizer = SpeechSynthesizer(
            input_dim=self.config["feature_extraction"]["temporal_feature_dim"],
            n_mels=self.config["synthesis"]["n_mels"]
        ).to(self.device)
        
        self.vocoder = Vocoder(
            n_mels=self.config["synthesis"]["n_mels"]
        ).to(self.device)
        
        # Load pretrained weights if available
        self._load_models()
        
    def _load_models(self):
        """Load pretrained model weights."""
        models_dir = Path(self.config["paths"]["models"])
        
        # Try to load each model's weights
        for model_name, model in [
            ("visual_feature_extractor", self.visual_feature_extractor),
            ("temporal_encoder", self.temporal_encoder),
            ("speech_synthesizer", self.speech_synthesizer),
            ("vocoder", self.vocoder)
        ]:
            model_path = models_dir / f"{model_name}.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded weights for {model_name} from {model_path}")
            else:
                logger.warning(f"No pretrained weights found for {model_name}")
                
    def process_video(self, video_path: str) -> Optional[np.ndarray]:
        """Process a video file and generate speech.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Generated audio waveform as numpy array, or None if processing failed
        """
        try:
            # Step 1: Extract lip regions from video
            lip_regions = self._extract_lip_regions(video_path)
            if lip_regions is None:
                logger.error("Failed to extract lip regions from video")
                return None
                
            # Step 2: Extract visual features
            visual_features = self._extract_visual_features(lip_regions)
            
            # Step 3: Encode temporal sequence
            temporal_features, _ = self.temporal_encoder(visual_features.unsqueeze(0))
            
            # Step 4: Generate mel-spectrogram
            mel_output = self.speech_synthesizer(temporal_features)
            
            # Step 5: Generate waveform
            waveform = self.vocoder(mel_output.transpose(1, 2))
            
            return waveform.squeeze().cpu().detach().numpy()
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return None
            
    def _extract_lip_regions(self, video_path: str) -> Optional[torch.Tensor]:
        """Extract lip regions from video frames.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Tensor of lip regions (seq_len, channels, height, width) or None if failed
        """
        lip_regions = []
        
        with VideoProcessor(video_path, self.config["preprocessing"]["target_fps"]) as vp:
            for frame_idx, frame in vp.get_frames():
                # Detect face and landmarks
                landmarks = self.face_detector.get_facial_landmarks(frame)
                if landmarks is None:
                    logger.warning(f"No face detected in frame {frame_idx}")
                    continue
                    
                # Extract lip region
                lip_region = self.lip_extractor.extract_lip_region(frame, landmarks)
                lip_regions.append(lip_region)
                
        if not lip_regions:
            return None
            
        # Convert to tensor and normalize
        lip_regions = np.stack(lip_regions)
        lip_regions = torch.from_numpy(lip_regions).float().permute(0, 3, 1, 2) / 255.0
        
        return lip_regions
        
    def _extract_visual_features(self, lip_regions: torch.Tensor) -> torch.Tensor:
        """Extract visual features from lip regions.
        
        Args:
            lip_regions: Tensor of lip regions (seq_len, channels, height, width)
            
        Returns:
            Visual features (seq_len, feature_dim)
        """
        batch_size = 32  # Process in batches to save memory
        features = []
        
        with torch.no_grad():
            self.visual_feature_extractor.eval()
            
            for i in range(0, len(lip_regions), batch_size):
                batch = lip_regions[i:i+batch_size].to(self.device)
                features.append(self.visual_feature_extractor(batch))
                
        return torch.cat(features, dim=0)
        
if __name__ == "__main__":
    # Example usage
    lipsync = LipSyncAI()
    audio = lipsync.process_video("data/raw/sample_video.mp4")
    
    if audio is not None:
        # Save audio to file
        import soundfile as sf
        sf.write("data/samples/output_audio.wav", audio, 
                samplerate=lipsync.config["synthesis"]["sample_rate"])
        print("Audio generated successfully!")
