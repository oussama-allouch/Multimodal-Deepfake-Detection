#!/usr/bin/env python3
"""
COMPREHENSIVE DEEPFAKE DETECTION SYSTEM DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script performs a complete end-to-end diagnostic of the deepfake system.
It identifies WHERE predictions become constant and WHY.

Checks:
  1. Model architecture & input/output shapes
  2. Data pipeline (frames extraction & preprocessing)
  3. Prediction consistency (same prediction for different inputs?)
  4. Label mapping validation
  5. Frontend-backend data flow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from model_handler import (
    DeepfakeDetector, VideoDeepfakeDetector, 
    SEBlock, SpatialAttention, TemporalAttention, LabelSmoothingBCE
)
from config import UPLOAD_FOLDER

print("\n" + "="*80)
print("DEEPFAKE SYSTEM COMPREHENSIVE DIAGNOSIS")
print("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[DIAG] STEP 1: MODEL ARCHITECTURE & SHAPES")
print("─"*80)

detector = DeepfakeDetector(model_path='./models/images_model.keras')
video_detector = VideoDeepfakeDetector(model_path='./models/videos_model_final_v3.keras')

if detector.model_loaded:
    print(f"[✓] IMAGE MODEL LOADED")
    print(f"    Input shape : {detector.model.input_shape}")
    print(f"    Output shape: {detector.model.output_shape}")
    print(f"    Parameters  : {detector.model.count_params():,}")
else:
    print(f"[✗] IMAGE MODEL FAILED TO LOAD")

if video_detector.model_loaded:
    print(f"[✓] VIDEO MODEL LOADED")
    print(f"    Input shape : {video_detector.model.input_shape}")
    print(f"    Output shape: {video_detector.model.output_shape}")
    print(f"    Parameters  : {video_detector.model.count_params():,}")
    
    # Check for multiple outputs
    if isinstance(video_detector.model.output, (list, tuple)):
        print(f"    [!] Model has {len(video_detector.model.output)} OUTPUT TENSORS (logit + attention)")
    else:
        print(f"    [!] Model has 1 OUTPUT TENSOR (logit only)")
else:
    print(f"[✗] VIDEO MODEL FAILED TO LOAD")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TEST DATA PIPELINE: FRAME EXTRACTION & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[DIAG] STEP 2: VIDEO FRAME EXTRACTION & PREPROCESSING")
print("─"*80)

# Create a simple test video (6 frames, 3x3 pixel, alternating colors)
test_video_path = UPLOAD_FOLDER / "test_video_diag.mp4"
test_video_path.parent.mkdir(exist_ok=True)

print(f"[*] Creating test video: {test_video_path}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(test_video_path), fourcc, 2.0, (224, 224))

# Frame 0-2: RED (fake indicator)
for i in range(3):
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # Red channel only
    out.write(frame)

# Frame 3-5: BLUE (real indicator)
for i in range(3):
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # Blue channel only
    out.write(frame)

out.release()
print(f"[✓] Test video created")

# Extract frames
print(f"\n[*] Extracting frames from test video...")
frames = video_detector._extract_uniform_frames(str(test_video_path))
if frames:
    print(f"[✓] Extracted {len(frames)} frames")
    
    # Show frame statistics
    for i, (frame_rgb, timestamp) in enumerate(frames[:3]):
        print(f"  Frame {i}: shape={frame_rgb.shape}, min={frame_rgb.min()}, max={frame_rgb.max()}, "
              f"mean_r={frame_rgb[:,:,0].mean():.1f}, mean_g={frame_rgb[:,:,1].mean():.1f}, "
              f"mean_b={frame_rgb[:,:,2].mean():.1f}")
else:
    print(f"[✗] Failed to extract frames")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[DIAG] STEP 3: FRAME PREPROCESSING VALIDATION")
print("─"*80)

if frames:
    print(f"[*] Preprocessing first frame...")
    frame_rgb, _ = frames[0]
    preprocessed = video_detector._preprocess_frame(frame_rgb)
    print(f"    Original shape : {frame_rgb.shape}, dtype={frame_rgb.dtype}, "
          f"range=[{frame_rgb.min()}, {frame_rgb.max()}]")
    print(f"    Preprocessed  : {preprocessed.shape}, dtype={preprocessed.dtype}, "
          f"range=[{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
    
    # Check ImageNet normalization
    print(f"    ImageNet MEAN : {video_detector.IMAGENET_MEAN}")
    print(f"    ImageNet STD  : {video_detector.IMAGENET_STD}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL PREDICTION: TEST FOR SATURATION
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[DIAG] STEP 4: MODEL PREDICTION SATURATION TEST")
print("─"*80)

if video_detector.model_loaded and frames:
    print(f"[*] Building input tensor from {len(frames)} frames...")
    
    processed = []
    for frame_rgb, _ in frames:
        preprocessed = video_detector._preprocess_frame(frame_rgb)
        processed.append(preprocessed)
    
    video_tensor = np.expand_dims(np.stack(processed, axis=0), axis=0)
    print(f"    Input shape: {video_tensor.shape}")
    print(f"    Input dtype: {video_tensor.dtype}")
    print(f"    Input range: [{video_tensor.min():.4f}, {video_tensor.max():.4f}]")
    
    print(f"\n[*] Running inference...")
    outputs = video_detector.model.predict(video_tensor, verbose=0)
    
    print(f"    Output type : {type(outputs)}")
    if isinstance(outputs, (list, tuple)):
        print(f"    Outputs     : {len(outputs)} tensors")
        for i, out in enumerate(outputs):
            print(f"      [{i}] shape={out.shape}, dtype={out.dtype}, "
                  f"range=[{out.min():.4f}, {out.max():.4f}], values={out.flatten()[:5]}")
    else:
        print(f"    Output shape: {outputs.shape}, dtype={outputs.dtype}, "
              f"range=[{outputs.min():.4f}, {outputs.max():.4f}]")
    
    # Parse prediction
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
        logit = float(outputs[0][0][0])
        attn_weights = outputs[1][0].tolist()
    else:
        logit = float(outputs[0][0])
        attn_weights = [1.0 / video_detector.NUM_FRAMES] * video_detector.NUM_FRAMES
    
    fake_score = float(1.0 / (1.0 + np.exp(-logit)))
    real_score = 1.0 - fake_score
    
    print(f"\n[RESULT] Prediction Summary:")
    print(f"    Logit       : {logit:.6f}")
    print(f"    Fake score  : {fake_score:.6f}")
    print(f"    Real score  : {real_score:.6f}")
    print(f"    Prediction  : {'Fake' if fake_score > 0.5 else 'Real'}")
    print(f"    Attention   : {attn_weights}")
    
    # Check for saturation
    if abs(fake_score - 0.5) < 0.05:
        print(f"\n    [!] PREDICTION IS CENTERED (~0.5) — Model may be untrained!")
    elif fake_score > 0.9 or fake_score < 0.1:
        print(f"\n    [!] PREDICTION IS SATURATED — Model output is extreme")
    else:
        print(f"\n    [✓] Prediction looks reasonable")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FRAME-BY-FRAME CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[DIAG] STEP 5: FRAME-BY-FRAME PREDICTION CONSISTENCY")
print("─"*80)

if detector.model_loaded and frames:
    print(f"[*] Testing if IMAGE model gives different predictions for different frames...")
    
    predictions_list = []
    for i, (frame_rgb, _) in enumerate(frames[:5]):  # Test first 5 frames
        try:
            preprocessed, _ = detector.preprocess_image(frame_rgb)
            if preprocessed is not None:
                result = detector.predict(preprocessed)
                if result:
                    predictions_list.append((i, result['real_score'], result['fake_score']))
                    print(f"    Frame {i}: Real={result['real_score']:.4f}, Fake={result['fake_score']:.4f}")
        except Exception as e:
            print(f"    Frame {i}: ERROR - {e}")
    
    # Check consistency
    if predictions_list:
        real_scores = [p[1] for p in predictions_list]
        fake_scores = [p[2] for p in predictions_list]
        real_std = np.std(real_scores)
        fake_std = np.std(fake_scores)
        
        print(f"\n    [VARIANCE ANALYSIS]")
        print(f"    Real score std : {real_std:.6f}")
        print(f"    Fake score std : {fake_std:.6f}")
        
        if real_std < 0.01 and fake_std < 0.01:
            print(f"    [✗] PREDICTIONS ARE IDENTICAL ACROSS FRAMES!")
            print(f"        → Model may be broken or untrained")
        else:
            print(f"    [✓] Predictions vary across frames (good!)")

# Clean up
if test_video_path.exists():
    test_video_path.unlink()
    print(f"\n[*] Cleaned up test video")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80 + "\n")
