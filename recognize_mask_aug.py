#!/usr/bin/env python3
"""
Recognition against original + masked embeddings
- Detector/Landmarks: InsightFace RetinaFace
- Alignment: ArcFace 5-point to 112x112
- Embeddings: FaceNet (InceptionResnetV1 pretrained='vggface2')
- Compares query embedding to all stored embeddings per user; uses max similarity

Usage:
  python recognize_mask_aug.py --cam 0
  python recognize_mask_aug.py --image path/to/test.jpg
  python recognize_mask_aug.py --cam 0 --threshold 0.65
"""

import sys
import os
import argparse
import cv2
import numpy as np
import torch
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1

EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")

ARCFACE_5PTS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def align_by_5pts(image_bgr: np.ndarray, kps5: np.ndarray, out_size=(112, 112)) -> np.ndarray | None:
    """Align face using 5-point landmarks to ArcFace standard"""
    src = np.asarray(kps5, dtype=np.float32)
    if src.shape != (5, 2):
        return None
    dst = ARCFACE_5PTS_112.copy()
    if out_size != (112, 112):
        sx = out_size[0] / 112.0
        sy = out_size[1] / 112.0
        dst[:, 0] *= sx
        dst[:, 1] *= sy
    try:
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    except Exception:
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None
    aligned = cv2.warpAffine(image_bgr, M, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def detect_available_cameras(max_test=5, verbose=True):
    """Detect available cameras on the system"""
    import warnings
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    
    available_cameras = []
    if verbose:
        print(f"🔍 Detecting available cameras (testing 0-{max_test-1})...")
    
    old_verbosity = None
    try:
        old_verbosity = cv2.getLogLevel()
        if hasattr(cv2, 'LOG_LEVEL_ERROR'):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        elif hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(0)
    except (AttributeError, Exception):
        pass
    
    for i in range(max_test):
        cap = None
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available_cameras.append(i)
                    if verbose:
                        print(f"  ✓ Camera {i}: Available")
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
            time.sleep(0.05)
    
    try:
        if old_verbosity is not None and hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(old_verbosity)
    except (AttributeError, Exception):
        pass
    
    if not available_cameras:
        if verbose:
            print("⚠️  No cameras detected!")
    else:
        if verbose:
            print(f"✓ Found {len(available_cameras)} camera(s): {available_cameras}")
    
    return available_cameras

def initialize_camera_with_fallback(camera_index=0, backends=[cv2.CAP_DSHOW, cv2.CAP_ANY]):
    """Initialize camera with automatic fallback if primary fails"""
    old_verbosity = None
    try:
        old_verbosity = cv2.getLogLevel()
        if hasattr(cv2, 'LOG_LEVEL_ERROR'):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        elif hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(0)
    except (AttributeError, Exception):
        pass
    
    cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    try:
                        if old_verbosity is not None and hasattr(cv2, 'setLogLevel'):
                            cv2.setLogLevel(old_verbosity)
                    except (AttributeError, Exception):
                        pass
                    return cap
                else:
                    if cap is not None:
                        cap.release()
                        cap = None
        except Exception:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
                cap = None
            continue
    
    try:
        if old_verbosity is not None and hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(old_verbosity)
    except (AttributeError, Exception):
        pass
    return None

def load_user_embeddings(user_dir: str) -> tuple[list[np.ndarray], dict]:
    """Load all embeddings (original + masked) for a user"""
    embs = []
    meta = {}
    
    # Load originals
    orig_dir = os.path.join(user_dir, 'originals')
    if os.path.exists(orig_dir):
        for f in sorted(os.listdir(orig_dir)):
            if f.endswith('.npy'):
                emb_path = os.path.join(orig_dir, f)
                try:
                    embs.append(np.load(emb_path))
                except Exception as e:
                    print(f"  ⚠️ Could not load {emb_path}: {e}")
    
    # Load masked embeddings
    masked_dir = os.path.join(user_dir, 'masked')
    if os.path.exists(masked_dir):
        for img_sub in sorted(os.listdir(masked_dir)):
            sub = os.path.join(masked_dir, img_sub)
            if not os.path.isdir(sub):
                continue
            for f in sorted(os.listdir(sub)):
                if f.endswith('.npy'):
                    emb_path = os.path.join(sub, f)
                    try:
                        embs.append(np.load(emb_path))
                    except Exception as e:
                        print(f"  ⚠️ Could not load {emb_path}: {e}")
    
    # Load metadata
    meta_path = os.path.join(user_dir, 'metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            print(f"  ⚠️ Could not load metadata: {e}")
    
    return embs, meta

def build_gallery() -> list[dict]:
    """Build gallery of all enrolled users with their embeddings"""
    users = []
    if not os.path.exists(EMBEDDINGS_DIR):
        return users
    
    print("📚 Loading enrolled users...")
    for d in sorted(os.listdir(EMBEDDINGS_DIR)):
        user_dir = os.path.join(EMBEDDINGS_DIR, d)
        if not os.path.isdir(user_dir):
            continue
        
        try:
            # Parse user_id and user_name from directory name (format: {id}_{name})
            parts = d.split('_', 1)
            user_id = parts[0]
            user_name = parts[1] if len(parts) > 1 else d
        except Exception:
            user_id = d
            user_name = d
        
        embs, meta = load_user_embeddings(user_dir)
        if not embs:
            print(f"  ⚠️ No embeddings found for {user_name} (ID: {user_id})")
            continue
        
        threshold = float(meta.get('recommended_threshold', 0.7))
        users.append({
            'user_id': user_id,
            'user_name': user_name,
            'dir': user_dir,
            'embeddings': embs,
            'threshold': threshold
        })
        print(f"  ✓ {user_name} (ID: {user_id}): {len(embs)} embeddings, threshold={threshold:.3f}")
    
    return users

def main():
    parser = argparse.ArgumentParser(description='Recognition (original + masked embeddings)')
    parser.add_argument('--cam', type=int, default=None, help='Camera index (default: 0 if no --image provided)')
    parser.add_argument('--image', type=str, default=None, help='Path to a test image (if provided, run once)')
    parser.add_argument('--det-size', type=int, nargs=2, default=(640, 640), help='Detector input size')
    parser.add_argument('--threshold', type=float, default=None, help='Global similarity threshold (overrides per-user thresholds if set)')
    parser.add_argument('--list-cameras', action='store_true', help='List available cameras and exit')
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        detect_available_cameras()
        return 0

    print("="*70)
    print("RECOGNITION WITH MASK AUGMENTATION")
    print("="*70)
    print("Models used:")
    print("  • InsightFace RetinaFace (auto-downloads on first use)")
    print("  • VGGFace2 InceptionResnetV1 (auto-downloads on first use)")
    print("="*70)

    # Build gallery
    gallery = build_gallery()
    if not gallery:
        print("\n❌ No enrolled users found in embeddings/")
        print("   Please register users first with: python register_mask_aug.py")
        return 1
    
    total_vectors = sum(len(u['embeddings']) for u in gallery)
    print(f"\n✓ Loaded {len(gallery)} users, {total_vectors} total embeddings")

    # Initialize detector and embedder
    print("\n🔧 Initializing models...")
    print("  • InsightFace RetinaFace (downloading if needed)...")
    try:
        detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        detector.prepare(ctx_id=0, det_size=tuple(args.det_size))
        print("    ✓ RetinaFace ready")
    except Exception as e:
        print(f"    ❌ Failed to initialize RetinaFace: {e}")
        return 1
    
    print("  • VGGFace2 (downloading if needed)...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device).eval()
        print(f"    ✓ VGGFace2 ready on {device}")
    except Exception as e:
        print(f"    ❌ Failed to initialize VGGFace2: {e}")
        return 1

    def embed_bgr(image_bgr: np.ndarray) -> np.ndarray | None:
        """Extract embedding from BGR image"""
        faces = detector.get(image_bgr)
        if not faces:
            return None
        best = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)))
        kps = getattr(best, 'kps', None)
        if kps is None or len(kps) != 5:
            return None
        aligned = align_by_5pts(image_bgr, kps, out_size=(112, 112))
        if aligned is None:
            return None
        x = aligned.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        t = torch.from_numpy(x).to(device)
        t = (t - 0.5) / 0.5
        with torch.no_grad():
            e = model(t)
            e = torch.nn.functional.normalize(e, p=2, dim=1)
        v = e.squeeze(0).cpu().numpy().astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

    def identify(emb: np.ndarray) -> tuple[str, str, float, float]:
        """Identify person from embedding"""
        best_user = None
        best_name = None
        best_score = -1.0
        used_threshold = args.threshold if args.threshold is not None else 0.65
        
        for u in gallery:
            # Use user-specific threshold if global threshold not set
            if args.threshold is None:
                # Use user's threshold, but cap it at 0.82 for more reasonable matching
                # (0.9 is too strict for real-world variations like lighting, beard, etc.)
                user_thresh = float(u.get('threshold', 0.65))
                thr = min(user_thresh, 0.82)  # Cap at 0.82 for better recognition
            else:
                thr = args.threshold
            
            # Compute similarities to all embeddings for this user
            sims = [float(np.dot(emb, g)) for g in u['embeddings']]
            smax = max(sims) if sims else -1.0
            
            if smax > best_score:
                best_score = smax
                best_user = u['user_id']
                best_name = u['user_name']
                used_threshold = thr
        
        return best_user, best_name, best_score, used_threshold

    # Single image mode (takes priority over camera)
    if args.image:
        print(f"\n📷 Processing image: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            print(f"❌ Could not read image: {args.image}")
            return 1
        
        emb = embed_bgr(img)
        if emb is None:
            print("❌ No face detected or embedding extraction failed")
            return 1
        
        uid, uname, score, thr = identify(emb)
        print("\n" + "="*70)
        if score >= thr:
            print(f"✅ MATCH: {uname} (ID={uid})")
            print(f"   Similarity: {score:.4f} (threshold: {thr:.4f})")
        else:
            print(f"❌ NO MATCH")
            print(f"   Best candidate: {uname} (ID={uid})")
            print(f"   Similarity: {score:.4f} (threshold: {thr:.4f})")
        print("="*70)
        return 0

    # Webcam mode (default to camera 0 if no image provided and no cam specified)
    if args.image is None:
        camera_index = args.cam if args.cam is not None else 0
        
        # Detect available cameras
        available_cameras = detect_available_cameras(verbose=False)
        if not available_cameras:
            print("❌ No cameras available. Please connect a camera and try again.")
            return 1
        
        # Validate camera index
        if camera_index not in available_cameras:
            print(f"⚠️  Camera {camera_index} not available. Available cameras: {available_cameras}")
            camera_index = available_cameras[0]
            print(f"   Using camera {camera_index} instead")
        
        print(f"\n📹 Opening camera {camera_index}...")
        cap = initialize_camera_with_fallback(camera_index)
        if cap is None:
            print(f"❌ Could not open camera {camera_index}")
            return 1
        
        current_camera_index = camera_index
        camera_detection_time = time.time()
        
        print("✓ Camera opened")
        print("\nControls:")
        print("  • ESC: Exit")
        print("  • SPACE: Force recognition (if face detected)")
        if len(available_cameras) > 1:
            print("  • C: Switch camera")
        print("-" * 70)
        
        last_recognition_time = 0
        recognition_interval = 0.5  # Recognize every 0.5 seconds
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("⚠️ Multiple camera read failures, attempting to reconnect...")
                    cap.release()
                    time.sleep(0.5)
                    cap = initialize_camera_with_fallback(current_camera_index)
                    if cap is None:
                        print(f"❌ Camera {current_camera_index} reconnection failed.")
                        break
                    consecutive_failures = 0
                time.sleep(0.02)
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            
            current_time = time.time()
            should_recognize = (current_time - last_recognition_time) >= recognition_interval
            
            # Extract embedding
            emb = embed_bgr(frame)
            
            msg = "No face detected"
            color = (0, 0, 255)
            status_color = (100, 100, 100)
            
            if emb is not None:
                if should_recognize:
                    uid, uname, score, thr = identify(emb)
                    last_recognition_time = current_time
                    
                    if score >= thr:
                        msg = f"MATCH: {uname} ({uid})"
                        color = (0, 255, 0)
                        status_color = (0, 255, 0)
                    else:
                        msg = f"NO MATCH: {uname} {score:.3f}/{thr:.3f}"
                        color = (0, 0, 255)
                        status_color = (0, 165, 255)
                else:
                    msg = "Processing..."
                    color = (0, 255, 255)
                    status_color = (0, 255, 255)
            
            # Draw on frame
            try:
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Users: {len(gallery)} | Embeddings: {total_vectors}", 
                           (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                if len(available_cameras) > 1:
                    cv2.putText(frame, f"Camera: {current_camera_index} (Press C to switch)", 
                               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("Recognition (ESC to exit)", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE - force recognition
                    if emb is not None:
                        uid, uname, score, thr = identify(emb)
                        last_recognition_time = current_time
                        if score >= thr:
                            print(f"✅ {uname} (ID={uid}) - Score: {score:.4f}")
                        else:
                            print(f"❌ NO MATCH - Best: {uname} - Score: {score:.4f}/{thr:.4f}")
                elif (key == ord('c') or key == ord('C')) and len(available_cameras) > 1:  # Switch camera
                    # Re-detect cameras if it's been more than 10 seconds
                    if time.time() - camera_detection_time > 10:
                        print("🔄 Re-detecting available cameras...")
                        available_cameras = detect_available_cameras(verbose=False)
                        camera_detection_time = time.time()
                    
                    if len(available_cameras) > 1:
                        try:
                            current_idx = available_cameras.index(current_camera_index)
                            next_idx = (current_idx + 1) % len(available_cameras)
                            new_camera_index = available_cameras[next_idx]
                        except ValueError:
                            new_camera_index = available_cameras[0]
                        
                        print(f"🔄 Switching to camera {new_camera_index}...")
                        cap.release()
                        time.sleep(0.3)
                        cap = initialize_camera_with_fallback(new_camera_index)
                        if cap is not None:
                            current_camera_index = new_camera_index
                            print(f"✓ Switched to camera {current_camera_index}")
                        else:
                            print(f"❌ Failed to switch to camera {new_camera_index}")
                            # Try to reconnect to original camera
                            cap = initialize_camera_with_fallback(current_camera_index)
                            if cap is None:
                                print("❌ Failed to reconnect to original camera!")
                                break
            except Exception as e:
                # Headless mode - print to console
                print(msg)
                time.sleep(0.1)
        
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("\n✓ Recognition stopped")
        return 0
    
    # Should not reach here if logic is correct
    print("\n❌ Please provide either --image <path> or --cam <index>")
    print("   Example: python recognize_mask_aug.py --cam 0")
    print("   Example: python recognize_mask_aug.py --image test.jpg")
    print("   Example: python recognize_mask_aug.py  (defaults to camera 0)")
    return 1

if __name__ == "__main__":
    code = main()
    sys.exit(code)

