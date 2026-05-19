from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration - all values loaded from .env file"""
    
    # API Settings
    API_TITLE: str
    API_VERSION: str
    API_HOST: str
    API_PORT: int
    
    # Detection Thresholds - Tier 1 (Spatial Gate)
    EDGE_THRESHOLD: float
    COLOR_QUANT_THRESHOLD: float
    
    # Voting Thresholds (Multi-Signal Consensus)
    EDGE_VOTE_THRESHOLD: float
    GRID_VOTE_THRESHOLD: float
    MVAD_VOTE_THRESHOLD: float
    BRISQUE_VOTE_THRESHOLD: float
    
    # Detection Thresholds - Tier 2 (ML)
    MVAD_BLOCKINESS_WEIGHT: float
    MVAD_PIXELATION_WEIGHT: float
    
    # Composite Score Weights
    BOUNDARY_EDGE_WEIGHT: float
    COLOR_QUANT_WEIGHT: float
    BRISQUE_WEIGHT: float
    
    # Alert Thresholds
    IMAGE_ALERT_THRESHOLD: float
    VIDEO_WINDOW_SIZE: int
    VIDEO_MIN_FLAGGED_FRAMES: int
    
    # Temporal Refinement - Video Only
    SCENE_CUT_SAD_THRESHOLD: float
    BLOCK_PERSISTENCE_FRAMES: int
    QP_THRESHOLD: int
    COLOR_SHIFT_SATURATION_DELTA: float
    
    # Grid Detection (FFmpeg blockdetect equivalent)
    GRID_SIZES: list = [8, 16, 32]
    
    # Model Settings
    MVAD_MODEL_PATH: Optional[str]
    USE_INCEPTION_FALLBACK: bool
    DEVICE: str
    
    # Performance
    MAX_WORKERS: int
    FRAME_BATCH_SIZE: int
    
    # Redis (for distributed stream state)
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    USE_REDIS: bool
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
