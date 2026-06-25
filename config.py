import os

import torch


class Config:
    # Paths. Environment variables let the released code run outside the author's workstation.
    MESH_ROOT = os.getenv("STDF_MESH_ROOT", r"F:\DataSetF\FaMoS")
    SKETCH_ROOT = os.getenv("STDF_SKETCH_ROOT", "./output_sketches_all")
    TEXT_JSON_DIR = os.getenv("STDF_TEXT_JSON_DIR", "./output_texts_all")
    CHECKPOINT_DIR = os.getenv("STDF_CHECKPOINT_DIR", "./bestmodel")
    OUTPUT_DIR = os.getenv("STDF_OUTPUT_DIR", "./outputs")

    # Reproducible subject-independent split.
    SPLIT_SEED = int(os.getenv("STDF_SPLIT_SEED", "42"))
    SPLIT_RATIOS = (0.8, 0.1, 0.1)

    # Abnormal sequences removed before training/evaluation.
    EXCLUDE_SEQUENCES = {
        "FaMoS_subject_004": ["anger"],
        "FaMoS_subject_015": ["lip_corners_down"],
        "FaMoS_subject_018": ["mouth_open"],
        "FaMoS_subject_028": ["surprise"],
        "FaMoS_subject_030": ["head_rotation_left_right"],
        "FaMoS_subject_031": ["mouth_open"],
        "FaMoS_subject_036": ["mouth_side"],
        "FaMoS_subject_041": ["head_rotation_left_right", "head_rotation_up_down", "rolling_lips"],
        "FaMoS_subject_053": ["mouth_extreme"],
        "FaMoS_subject_064": ["head_rotation_left_right"],
        "FaMoS_subject_065": ["surprise"],
        "FaMoS_subject_069": ["mouth_side"],
        "FaMoS_subject_087": ["mouth_side"],
        "FaMoS_subject_091": ["head_rotation_up_down"],
        "FaMoS_subject_060": ["jaw"],
    }

    # Data and model.
    SEQ_LENGTH = int(os.getenv("STDF_SEQ_LENGTH", "160"))
    IMAGE_SIZE = int(os.getenv("STDF_IMAGE_SIZE", "256"))
    N_VERTICES = int(os.getenv("STDF_N_VERTICES", "5023"))
    HIDDEN_DIM = int(os.getenv("STDF_HIDDEN_DIM", "256"))
    N_BLOCKS = int(os.getenv("STDF_N_BLOCKS", "6"))
    N_HEADS = int(os.getenv("STDF_N_HEADS", "4"))
    K_EIG = int(os.getenv("STDF_K_EIG", "128"))
    TEXT_EMBED_DIM = int(os.getenv("STDF_TEXT_EMBED_DIM", "512"))
    SKETCH_EMBED_DIM = int(os.getenv("STDF_SKETCH_EMBED_DIM", "512"))
    SKETCH_BACKBONE = os.getenv("STDF_SKETCH_BACKBONE", "resnet50")
    CLIP_MODEL = os.getenv("STDF_CLIP_MODEL", "openai/clip-vit-base-patch32")

    # Reviewer-facing ablation switches.
    USE_GCTD = os.getenv("STDF_USE_GCTD", "1") != "0"
    USE_MIXSTYLE = os.getenv("STDF_USE_MIXSTYLE", "1") != "0"
    USE_DUAL_PATH_ATTENTION = os.getenv("STDF_USE_DUAL_PATH_ATTENTION", "1") != "0"
    USE_SPECTRAL_DIFFUSION = os.getenv("STDF_USE_SPECTRAL_DIFFUSION", "1") != "0"

    # Training.
    BATCH_SIZE = int(os.getenv("STDF_BATCH_SIZE", "4"))
    LR = float(os.getenv("STDF_LR", "1e-4"))
    WEIGHT_DECAY = float(os.getenv("STDF_WEIGHT_DECAY", "1e-2"))
    EPOCHS = int(os.getenv("STDF_EPOCHS", "200"))
    NUM_WORKERS = int(os.getenv("STDF_NUM_WORKERS", "0"))
    DEVICE = os.getenv("STDF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Composite objective weights described in the manuscript.
    LAMBDA_FLOW = float(os.getenv("STDF_LAMBDA_FLOW", "1.0"))
    LAMBDA_ID = float(os.getenv("STDF_LAMBDA_ID", "5.0"))
    LAMBDA_SEM = float(os.getenv("STDF_LAMBDA_SEM", "0.5"))
    LAMBDA_TEMP = float(os.getenv("STDF_LAMBDA_TEMP", "0.1"))
    LAMBDA_EDGE = float(os.getenv("STDF_LAMBDA_EDGE", "0.05"))

    # Inference/evaluation.
    DEFAULT_ODE_STEPS = int(os.getenv("STDF_DEFAULT_ODE_STEPS", "32"))
    INFERENCE_FRAME_COUNT = int(os.getenv("STDF_INFERENCE_FRAME_COUNT", str(SEQ_LENGTH)))
    INFERENCE_SEED = int(os.getenv("STDF_INFERENCE_SEED", "1234"))
    EXPORT_DENOISE_TRAJECTORY = os.getenv("STDF_EXPORT_DENOISE_TRAJECTORY", "0") != "0"
    DEFAULT_SOLVER = os.getenv("STDF_DEFAULT_SOLVER", "euler")

    @classmethod
    def ensure_dirs(cls) -> None:
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
