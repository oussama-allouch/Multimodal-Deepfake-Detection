#!/usr/bin/env python3
"""
Comprehensive test script to verify the video deepfake detection pipeline.
Tests frame extraction, preprocessing, model predictions, and per-frame variation.
"""

import sys
sys.path.insert(0, '/vercel/share/v0-project/backend')

import numpy as np
import cv2
from pathlib import Path
import tempfile
from model_handler import get_video_detector, DeepfakeVideoDetector

def create_test_video(output_path, width=320, height=240, fps=30, duration_sec=2):
    """Create a simple test video with moving frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    num_frames = int(fps * duration_sec)
    for i in range(num_frames):
        # Create frame with changing patterns to ensure frames are different
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving shapes and text
        cv2.rectangle(frame, (50 + i*2, 50), (150 + i*2, 150), (0, 255, 0), -1)
        cv2.circle(frame, (100 + i, 100), 30 + i % 20, (255, 0, 0), -1)
        cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Test video created: {output_path} ({num_frames} frames @ {fps}fps)")
    return output_path

def test_frame_extraction():
    """Test uniform frame extraction with duplicate detection."""
    print("\n" + "="*80)
    print("TEST 1: Frame Extraction & Duplicate Detection")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        create_test_video(video_path)
        
        detector = get_video_detector()
        frames = detector._extract_uniform_frames(str(video_path))
        
        if frames is None:
            print("✗ FAILED: Could not extract frames")
            return False
        
        print(f"✓ Extracted {len(frames)} frames")
        
        # Check frame uniqueness
        frame_stats = []
        for i, (frame, ts) in enumerate(frames):
            frame_stats.append({
                'index': i,
                'timestamp': ts,
                'shape': frame.shape,
                'dtype': frame.dtype,
                'mean': frame.mean(),
                'std': frame.std(),
                'hash': hash(frame.tobytes()[:1000])
            })
        
        unique_hashes = len(set(s['hash'] for s in frame_stats))
        print(f"✓ Frame uniqueness: {unique_hashes}/{len(frames)} unique")
        
        if unique_hashes < len(frames) * 0.8:
            print("✗ WARNING: Too many duplicate frames detected!")
            return False
        
        return True

def test_preprocessing():
    """Test preprocessing pipeline for consistency."""
    print("\n" + "="*80)
    print("TEST 2: Preprocessing & Normalization")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        create_test_video(video_path)
        
        detector = get_video_detector()
        frames = detector._extract_uniform_frames(str(video_path))
        
        processed_frames = []
        for i, (frame_rgb, _) in enumerate(frames):
            proc = detector._preprocess_frame(frame_rgb)
            processed_frames.append(proc)
            print(f"  Frame {i}: shape={proc.shape}, dtype={proc.dtype}, "
                  f"mean={proc.mean():.6f}, range=[{proc.min():.6f}, {proc.max():.6f}]")
        
        # Check consistency
        print(f"\n✓ All frames preprocessed successfully")
        print(f"  Shape consistency: {all(p.shape == (224, 224, 3) for p in processed_frames)}")
        print(f"  Dtype consistency: {all(p.dtype == np.float32 for p in processed_frames)}")
        
        # Check for duplicates in preprocessed frames
        proc_hashes = [hash(p.tobytes()[:1000]) for p in processed_frames]
        unique_proc = len(set(proc_hashes))
        print(f"  Preprocessed uniqueness: {unique_proc}/{len(processed_frames)} unique")
        
        return True

def test_model_predictions():
    """Test model predictions with per-frame variation."""
    print("\n" + "="*80)
    print("TEST 3: Model Predictions & Per-Frame Variation")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        create_test_video(video_path)
        
        detector = get_video_detector()
        
        if not detector.model_loaded:
            print("✗ CRITICAL: Video model not loaded")
            return False
        
        result = detector.process_video(str(video_path))
        
        if result is None:
            print("✗ CRITICAL: process_video returned None")
            return False
        
        print(f"\n✓ Video processed successfully")
        print(f"\nAggregated Results:")
        print(f"  Prediction: {result['aggregated']['prediction']}")
        print(f"  Fake score: {result['aggregated']['fake_score']:.6f}")
        print(f"  Real score: {result['aggregated']['real_score']:.6f}")
        print(f"  Confidence: {result['aggregated']['confidence']:.6f}")
        print(f"  Logit: {result['aggregated'].get('logit', 'N/A')}")
        
        # Check per-frame variation
        frame_preds = result['frame_predictions']
        fake_scores = [f['fake_score'] for f in frame_preds]
        real_scores = [f['real_score'] for f in frame_preds]
        
        print(f"\nPer-Frame Statistics:")
        print(f"  Total frames: {len(frame_preds)}")
        print(f"  Fake scores range: [{min(fake_scores):.6f}, {max(fake_scores):.6f}]")
        print(f"  Real scores range: [{min(real_scores):.6f}, {max(real_scores):.6f}]")
        print(f"  Unique fake scores: {len(set(round(f, 6) for f in fake_scores))}/{len(fake_scores)}")
        
        # Critical check: are per-frame predictions varying?
        unique_fake_rounded = len(set(round(f, 4) for f in fake_scores))
        
        if unique_fake_rounded == 1:
            print(f"\n✗ CRITICAL: All per-frame predictions are IDENTICAL!")
            print(f"  This indicates attention weights are not being used correctly.")
            print(f"  OR the attention weights are all equal/near-equal.")
            return False
        else:
            print(f"\n✓ Per-frame predictions VARY ({unique_fake_rounded} unique values)")
            print(f"  This is CORRECT behavior.")
        
        # Print first and last few frames
        print(f"\nFrame-by-frame breakdown (first 3 + last 2):")
        for f in frame_preds[:3]:
            print(f"  Frame {f['frame_index']}: fake={f['fake_score']:.4f}, "
                  f"real={f['real_score']:.4f}, attn={f.get('attention_weight', 0):.4f}")
        if len(frame_preds) > 5:
            print(f"  ... ({len(frame_preds) - 5} more) ...")
        for f in frame_preds[-2:]:
            print(f"  Frame {f['frame_index']}: fake={f['fake_score']:.4f}, "
                  f"real={f['real_score']:.4f}, attn={f.get('attention_weight', 0):.4f}")
        
        return True

def test_no_nan_values():
    """Test that no NaN values appear in results."""
    print("\n" + "="*80)
    print("TEST 4: NaN Value Detection")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        create_test_video(video_path)
        
        detector = get_video_detector()
        result = detector.process_video(str(video_path))
        
        if result is None:
            print("✗ process_video returned None")
            return False
        
        # Check aggregated
        agg = result['aggregated']
        nan_count = 0
        for key, val in agg.items():
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                print(f"✗ NaN/Inf in aggregated[{key}]: {val}")
                nan_count += 1
            elif isinstance(val, list):
                for i, v in enumerate(val):
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        print(f"✗ NaN/Inf in aggregated[{key}][{i}]: {v}")
                        nan_count += 1
        
        # Check per-frame
        for i, frame in enumerate(result['frame_predictions']):
            for key, val in frame.items():
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    print(f"✗ NaN/Inf in frame[{i}][{key}]: {val}")
                    nan_count += 1
        
        if nan_count > 0:
            print(f"✗ Found {nan_count} NaN/Inf values")
            return False
        else:
            print(f"✓ No NaN/Inf values detected")
            return True

def main():
    print("\n" + "█"*80)
    print("█ VIDEO DEEPFAKE DETECTION PIPELINE - COMPREHENSIVE TEST")
    print("█"*80)
    
    tests = [
        ("Frame Extraction", test_frame_extraction),
        ("Preprocessing", test_preprocessing),
        ("Model Predictions", test_model_predictions),
        ("NaN Detection", test_no_nan_values),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"\n✗ TEST EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "ERROR"
    
    print("\n" + "█"*80)
    print("█ TEST SUMMARY")
    print("█"*80)
    for test_name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {test_name}: {result}")
    
    all_pass = all(r == "PASS" for r in results.values())
    print("\n" + ("█"*80 if all_pass else "!"*80))
    print("█ OVERALL: " + ("ALL TESTS PASSED ✓" if all_pass else "SOME TESTS FAILED ✗"))
    print("█"*80)
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
