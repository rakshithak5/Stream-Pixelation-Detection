"""
Tier 2 - ML verification models
MVAD (primary) with InceptionV3 fallback
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from src.core.config import settings


class MVADWrapper:
    """
    Wrapper for MVAD model (ChenFeng-Bristol/MVAD)
    RMViT architecture with temporal context support
    """

    DEFAULT_CHECKPOINT = Path("MVAD_repo/logs/checkpoints/mvad.ckpt")
    ARTEFACTS = [
        'motion_blur', 'dark_scenes', 'graininess', 'aliasing', 'banding',
        'blockiness', 'spatial_blur', 'frame_drop', 'transmission_error',
        'black_screen'
    ]

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path or settings.MVAD_MODEL_PATH

        if self.model_path is None and self.DEFAULT_CHECKPOINT.exists():
            self.model_path = str(self.DEFAULT_CHECKPOINT)

        if self.model_path:
            try:
                self._load_mvad_model()
            except Exception as e:
                print(f"Failed to load MVAD model: {e}")
                self.model = None

        if self.model is None:
            print("⚠ MVAD model unavailable; MVAD inference will be skipped.")

    def _load_mvad_model(self):
        """
        Load the pretrained MVAD artefact detection model from checkpoint.
        """
        import sys
        
        # Add MVAD_repo to path so it can import its own modules
        # MVAD_repo is at project root, not in src/models/
        project_root = Path(__file__).resolve().parent.parent.parent
        mvad_root = project_root / "MVAD_repo"
        
        if not mvad_root.exists():
            raise FileNotFoundError(f"MVAD_repo not found at {mvad_root}")
        
        # Add MVAD_repo to path PERMANENTLY (don't remove)
        if str(mvad_root) not in sys.path:
            sys.path.insert(0, str(mvad_root))
        
        # Now import from the MVAD models package
        from models.artefact_net import ArtefactNet

        ckpt_path = Path(self.model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MVAD checkpoint not found at {ckpt_path}")

        self.model = ArtefactNet(
            artefacts=self.ARTEFACTS,
            feat_dim=768,
            head_dim=64,
            head_dropout=0.5,
            pretrained_backbone_path=None
        )

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get('state_dict', checkpoint)
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"MVAD checkpoint loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, frame: np.ndarray, temporal_context: Optional[list] = None) -> Tuple[float, float]:
        """
        Run MVAD inference.

        Args:
            frame: Current frame (H, W, 3)
            temporal_context: List of surrounding frames for video path

        Returns:
            (blockiness_score, pixelation_score) both in [0, 1]
        """
        if self.model is None:
            return (0.0, 0.0)

        input_tensor = self._prepare_input(frame, temporal_context)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            outputs, _ = self.model(input_tensor)

        # MVAD outputs logits for 10 artifact types
        # Apply sigmoid to convert to probabilities [0, 1]
        
        # Blockiness-related artifacts
        blockiness = torch.sigmoid(outputs['blockiness']).item()
        banding = torch.sigmoid(outputs['banding']).item()
        
        # Pixelation-related artifacts
        spatial_blur = torch.sigmoid(outputs['spatial_blur']).item()
        aliasing = torch.sigmoid(outputs['aliasing']).item()
        
        # Combine related artifacts for better sensitivity
        # Blockiness: use max of blockiness and banding (often occur together)
        mvad_blockiness = max(blockiness, banding * 0.7, (blockiness + banding) / 2)
        
        # Pixelation: combine blur and aliasing
        mvad_pixelation = max(spatial_blur, aliasing * 0.8, (spatial_blur + aliasing) / 2)
        
        return (mvad_blockiness, mvad_pixelation)

    def _prepare_input(self, frame: np.ndarray, temporal_context: Optional[list] = None) -> torch.Tensor:
        """Prepare a 5D tensor for MVAD input (B, C, T, H, W)."""
        import cv2

        target_frames = 4
        frames = [frame]
        if temporal_context:
            frames.extend(temporal_context[: target_frames - 1])

        while len(frames) < target_frames:
            frames.append(frames[-1])

        processed = []
        for f in frames[:target_frames]:
            # Ensure frame is BGR (3 channels)
            if len(f.shape) == 2:
                # Grayscale -> BGR
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif f.shape[2] == 4:
                # RGBA -> BGR (remove alpha channel)
                f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
            elif f.shape[2] != 3:
                raise ValueError(f"Unexpected number of channels: {f.shape[2]}")
            
            # Resize to 224x224
            f_resized = cv2.resize(f, (224, 224))
            
            # Normalize to [0, 1]
            f_norm = f_resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            f_norm = (f_norm - mean) / std
            
            # Convert to tensor (C, H, W)
            tensor = torch.from_numpy(f_norm.transpose(2, 0, 1)).float()
            processed.append(tensor)

        clip = torch.stack(processed, dim=1)  # (C, T, H, W)
        return clip.unsqueeze(0)


class InceptionV3Fallback(nn.Module):
    """
    InceptionV3 backbone with two output heads.
    Drop-in replacement for MVAD if needed.
    Fine-tune on your own labeled stream samples.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained InceptionV3
        self.backbone = models.inception_v3(pretrained=pretrained)
        
        # Remove original classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Two output heads
        self.blockiness_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.pixelation_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        blockiness = self.blockiness_head(features)
        pixelation = self.pixelation_head(features)
        return blockiness, pixelation


class ModelManager:
    """
    Manages ML models with automatic fallback.
    Tries MVAD first, falls back to InceptionV3 if unavailable.
    """
    
    def __init__(self):
        self.device = settings.DEVICE
        self.mvad = None
        self.inception = None
        
        # Try MVAD - REQUIRED, no fallback
        if settings.MVAD_MODEL_PATH:
            try:
                self.mvad = MVADWrapper(settings.MVAD_MODEL_PATH, self.device)
                if self.mvad.model is None:
                    raise RuntimeError("MVAD model could not be initialized")
                print("✓ MVAD model loaded successfully")
            except Exception as e:
                print(f"✗ MVAD load failed: {e}")
                print("⚠️  CRITICAL: MVAD is required. InceptionV3 fallback is DISABLED.")
                self.mvad = None
        
        # InceptionV3 fallback is DISABLED per user request
        # User wants MVAD only for accurate detection
    
    def predict(self, frame: np.ndarray, temporal_context: Optional[list] = None) -> Tuple[float, float]:
        """
        Run ML inference - MVAD ONLY (no fallback).
        
        Returns:
            (blockiness_score, pixelation_score)
        """
        # Use MVAD only - if not available, return zeros
        if self.mvad is not None and self.mvad.model is not None:
            return self.mvad.predict(frame, temporal_context)
        
        # No MVAD = no detection (InceptionV3 disabled per user request)
        print("⚠️  MVAD not available - returning neutral scores")
        return (0.0, 0.0)
    
    def _predict_inception(self, frame: np.ndarray) -> Tuple[float, float]:
        """Run InceptionV3 inference"""
        import cv2
        
        # Preprocess
        frame_resized = cv2.resize(frame, (299, 299))  # InceptionV3 input size
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame_normalized = (frame_normalized - mean) / std
        
        tensor = torch.from_numpy(frame_normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            blockiness, pixelation = self.inception(tensor)
        
        return (blockiness.item(), pixelation.item())


# Global model manager instance
model_manager = ModelManager()
