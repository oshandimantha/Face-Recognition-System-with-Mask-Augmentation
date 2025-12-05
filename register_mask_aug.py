#!/usr/bin/env python3
"""
Registration with synthetic mask augmentation (Option A)
- Detector/Landmarks: InsightFace RetinaFace
- Alignment: ArcFace 5-point affine to 112x112
- Embeddings: FaceNet (InceptionResnetV1 pretrained='vggface2')
- Mask augmentation: Local masktheface (6 mask types per original image)
- Output: 3 original + 6 masked embeddings per image = 9/image × 3 = 27 total

Usage:
  python register_mask_aug.py --name "doctor name"
  python register_mask_aug.py --name "john" --id "88"
"""

import sys
import os
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import json
import time
import torch
import dlib
from imutils import face_utils

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MASKTHEFACE_PATH = os.path.join(PROJECT_ROOT, 'masktheface')
MASKS_DIR = os.path.join(PROJECT_ROOT, 'masks')
if MASKTHEFACE_PATH not in sys.path:
    sys.path.insert(0, MASKTHEFACE_PATH)

# Import masktheface utilities
from masktheface.utils.aux_functions import (
    mask_face, get_six_points, shape_to_landmarks, rect_to_bb
)
from masktheface.utils.read_cfg import read_cfg

from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1

EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

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

def create_mask_visualization(original_img: np.ndarray, aligned_face: np.ndarray, 
                              masked_images: dict, mask_types: list, output_path: str):
    """Create a visualization showing original and all masked variants"""
    try:
        # Resize images for display (make them uniform size)
        display_size = (200, 200)
        
        # Resize original image (crop center if needed)
        orig_display = cv2.resize(original_img, display_size) if original_img.shape[:2] != display_size else original_img
        
        # Resize aligned face (convert from RGB to BGR for display)
        aligned_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR) if len(aligned_face.shape) == 3 else aligned_face
        aligned_display = cv2.resize(aligned_bgr, display_size) if aligned_bgr.shape[:2] != display_size else aligned_bgr
        
        # Create grid: 2 rows, 4 columns
        # Row 1: Original, Aligned, Mask1, Mask2
        # Row 2: Mask3, Mask4, Mask5, Mask6
        rows = []
        
        # First row: Original, Aligned, first 2 masks
        row1_images = [orig_display.copy(), aligned_display.copy()]
        for i, mtype in enumerate(mask_types[:2]):
            if mtype in masked_images:
                masked = masked_images[mtype]
                masked_resized = cv2.resize(masked, display_size) if masked.shape[:2] != display_size else masked
                row1_images.append(masked_resized.copy())
            else:
                row1_images.append(np.zeros((*display_size, 3), dtype=np.uint8))
        
        row1 = np.hstack(row1_images)
        
        # Second row: remaining 4 masks
        row2_images = []
        for mtype in mask_types[2:]:
            if mtype in masked_images:
                masked = masked_images[mtype]
                masked_resized = cv2.resize(masked, display_size) if masked.shape[:2] != display_size else masked
                row2_images.append(masked_resized.copy())
            else:
                row2_images.append(np.zeros((*display_size, 3), dtype=np.uint8))
        
        # Pad row2 to 4 images if needed
        while len(row2_images) < 4:
            row2_images.append(np.zeros((*display_size, 3), dtype=np.uint8))
        
        row2 = np.hstack(row2_images[:4])
        
        # Combine rows
        combined = np.vstack([row1, row2])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        labels_row1 = ["Original", "Aligned"] + [mtype.replace('_', ' ').title() for mtype in mask_types[:2]]
        labels_row2 = [mtype.replace('_', ' ').title() for mtype in mask_types[2:6]]
        
        # Add background rectangles for labels
        label_height = 25
        for i, label in enumerate(labels_row1):
            x = i * display_size[0]
            cv2.rectangle(combined, (x, 0), (x + display_size[0], label_height), bg_color, -1)
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x + (display_size[0] - text_size[0]) // 2
            cv2.putText(combined, label, (text_x, 18), font, font_scale, color, thickness)
        
        for i, label in enumerate(labels_row2):
            x = i * display_size[0]
            y = display_size[1] + label_height
            cv2.rectangle(combined, (x, y), (x + display_size[0], y + label_height), bg_color, -1)
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x + (display_size[0] - text_size[0]) // 2
            cv2.putText(combined, label, (text_x, y + 18), font, font_scale, color, thickness)
        
        # Add separator lines
        for i in range(1, 4):
            x = i * display_size[0]
            cv2.line(combined, (x, 0), (x, combined.shape[0]), (100, 100, 100), 1)
        
        cv2.line(combined, (0, display_size[1] + label_height), (combined.shape[1], display_size[1] + label_height), (100, 100, 100), 2)
        
        # Save visualization
        cv2.imwrite(output_path, combined)
        
    except Exception as e:
        print(f"    Visualization error: {e}")
        import traceback
        traceback.print_exc()

def create_summary_visualization(viz_paths: list, output_path: str):
    """Create a summary visualization showing all 3 images side by side"""
    try:
        images = []
        for path in viz_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
        
        if len(images) == 0:
            return
        
        # Resize all to same height (keep aspect ratio)
        target_height = 450
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized = cv2.resize(img, (new_w, target_height))
            resized_images.append(resized)
        
        # Combine horizontally
        combined = np.hstack(resized_images)
        
        # Add title
        title_height = 40
        title_img = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
        title_text = "Mask Augmentation Summary - All 3 Images"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
        text_x = (combined.shape[1] - text_size[0]) // 2
        cv2.putText(title_img, title_text, (text_x, 30), font, font_scale, (255, 255, 255), thickness)
        
        final = np.vstack([title_img, combined])
        cv2.imwrite(output_path, final)
        
    except Exception as e:
        print(f"    Summary visualization error: {e}")

def select_images_from_device(required_count: int = 3) -> list:
    """Open file picker to select multiple images"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG'),
            ('All files', '*.*')
        ]
        print(f"\n📸 Please multi-select {required_count} image(s) from your device (Ctrl/Shift-click)")
        selections = list(filedialog.askopenfilenames(
            title=f'Select {required_count} Face Images',
            filetypes=filetypes,
            initialdir=os.path.expanduser('~')
        ))
        if not selections:
            print("❌ Selection cancelled")
            return []
        valid = [p for p in selections if os.path.splitext(p)[1].lower() in ('.jpg', '.jpeg', '.png')]
        if len(valid) < required_count:
            messagebox.showerror("Invalid Selection", f"Please select at least {required_count} valid images.\n\nSelected: {len(valid)}", parent=root)
            return []
        if len(valid) > required_count:
            # Pick best by sharpness
            print(f"\nℹ️ You selected {len(valid)} images; picking the best {required_count} by sharpness...")
            scored = []
            for p in valid:
                try:
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    scored.append((blur, p))
                except Exception:
                    continue
            scored.sort(key=lambda x: x[0], reverse=True)
            valid = [p for _, p in scored[:required_count]]
        print(f"\n✓ Selected {len(valid)} image(s):")
        for i, img_path in enumerate(valid, 1):
            size_mb = os.path.getsize(img_path) / (1024 * 1024)
            print(f"   {i}. {os.path.basename(img_path)} ({size_mb:.2f} MB)")
        return valid
    finally:
        try:
            root.destroy()
        except:
            pass

def apply_mask_with_masktheface(image_bgr: np.ndarray, mask_type: str, dlib_detector, dlib_predictor) -> np.ndarray | None:
    """Apply mask using masktheface library with proper config path"""
    try:
        # Convert to RGB for masktheface
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect face with dlib
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face_locations = dlib_detector(gray, 1)
        if not face_locations:
            return None
        
        # Use first face
        face_location = face_locations[0]
        shape = dlib_predictor(gray, face_location)
        shape = face_utils.shape_to_np(shape)
        face_landmarks = shape_to_landmarks(shape)
        face_location_bb = rect_to_bb(face_location)
        
        # Get six points and angle
        six_points, angle = get_six_points(face_landmarks, image_rgb)
        
        # Create args object for masktheface
        class Args:
            def __init__(self):
                self.pattern = ""
                self.pattern_weight = 0.5
                self.color = ""
                self.color_weight = 0.5
                self.verbose = False
        
        args = Args()
        
        # Set the correct config path - masktheface looks for masks/masks.cfg relative to current dir
        # Our masks directory is at project root, so change to project root
        original_dir = os.getcwd()
        try:
            # Change to project root so masks/masks.cfg can be found
            os.chdir(PROJECT_ROOT)
            # Apply mask - masktheface will look for masks/masks.cfg relative to PROJECT_ROOT
            # When return_mask_status=False, mask_face returns only the image (not a tuple)
            masked_image = mask_face(
                image_rgb, face_location_bb, six_points, angle, args, type=mask_type, return_mask_status=False
            )
        finally:
            os.chdir(original_dir)
        
        if masked_image is None:
            return None
        
        # Convert back to BGR
        return cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"    Error applying {mask_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Register user with mask augmentation (Option A)')
    parser.add_argument('--name', type=str, default=None, help='User name (if omitted, GUI prompt)')
    parser.add_argument('--id', type=str, default=None, help='User ID (auto if omitted)')
    parser.add_argument('--mask-types', type=str, nargs='*', 
                        default=['surgical_blue', 'surgical_green', 'cloth', 'n95', 'kn95', 'gas'],
                        help='6 mask types to synthesize per image (using masktheface templates)')
    parser.add_argument('--det-size', type=int, nargs=2, default=(640, 640), help='Detector input size')
    parser.add_argument('--min-blur', type=float, default=40.0, help='Minimum Laplacian variance for quality filtering')
    args = parser.parse_args()

    if len(args.mask_types) != 6:
        print(f"⚠️ You provided {len(args.mask_types)} mask types; expecting 6. Proceeding anyway.")

    print("="*70)
    print("REGISTRATION WITH MASK AUGMENTATION (Option A)")
    print("="*70)
    print("Models used:")
    print("  • InsightFace RetinaFace (auto-downloads on first use)")
    print("  • VGGFace2 InceptionResnetV1 (auto-downloads on first use)")
    print("  • masktheface (local)")
    print("="*70)

    # Get user name
    if args.name:
        user_name = args.name.strip()
        print(f"\n✓ User name: {user_name}")
    else:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        user_name = simpledialog.askstring("User Name", "Enter the user name for registration:", parent=root)
        root.destroy()
        if not user_name:
            print("❌ No user name provided")
            return 1
        user_name = user_name.strip()
        print(f"\n✓ User name: {user_name}")

    # Select images
    image_files = select_images_from_device(required_count=3)
    if not image_files or len(image_files) != 3:
        print("\n❌ Need exactly 3 images")
        return 1

    # Generate user ID
    if args.id:
        user_id = args.id.strip()
        print(f"\n✓ User ID: {user_id}")
    else:
        existing = []
        if os.path.exists(EMBEDDINGS_DIR):
            for item in os.listdir(EMBEDDINGS_DIR):
                if os.path.isdir(os.path.join(EMBEDDINGS_DIR, item)):
                    try:
                        prefix = item.split('_')[0]
                        if prefix.isdigit():
                            existing.append(int(prefix))
                    except:
                        pass
        user_id = str((max(existing) + 1) if existing else 1)
        print(f"\n✓ Auto-generated User ID: {user_id}")

    # Setup directories
    user_dir = os.path.join(EMBEDDINGS_DIR, f"{user_id}_{user_name}")
    originals_dir = os.path.join(user_dir, "originals")
    masked_dir = os.path.join(user_dir, "masked")
    visualizations_dir = os.path.join(user_dir, "visualizations")
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    # Initialize models
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
    
    print("  • dlib for masktheface...")
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_model_path = os.path.join(MASKTHEFACE_PATH, "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(dlib_model_path):
        dlib_model_path = os.path.join(PROJECT_ROOT, "dlib_models", "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(dlib_model_path):
        print("    ⚠️ dlib model not found. Trying to download...")
        try:
            from masktheface.utils.aux_functions import download_dlib_model
            download_dlib_model()
            dlib_model_path = os.path.join(MASKTHEFACE_PATH, "dlib_models", "shape_predictor_68_face_landmarks.dat")
        except Exception as e:
            print(f"    ❌ Could not download dlib model: {e}")
            return 1
    
    if not os.path.exists(dlib_model_path):
        print(f"    ❌ dlib model not found at: {dlib_model_path}")
        return 1
    
    try:
        dlib_predictor = dlib.shape_predictor(dlib_model_path)
        print(f"    ✓ dlib predictor loaded from {dlib_model_path}")
    except Exception as e:
        print(f"    ❌ Could not load dlib predictor: {e}")
        return 1

    def embed_aligned(rgb112: np.ndarray) -> np.ndarray | None:
        """Extract embedding from aligned 112x112 RGB face"""
        try:
            x = rgb112.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]
            t = torch.from_numpy(x).to(device)
            t = (t - 0.5) / 0.5
            with torch.no_grad():
                e = model(t)
                e = torch.nn.functional.normalize(e, p=2, dim=1)
            v = e.squeeze(0).cpu().numpy().astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            return v
        except Exception as e:
            print(f"    Embedding error: {e}")
            return None

    quality_scores = []
    saved_embeddings = []
    metadata_images = []
    visualization_paths = []

    print(f"\n📁 Processing 3 original image(s) with mask augmentation...")
    print(f"   Expected: 3 originals + 6 masked × 3 = 27 total embeddings")
    print("-" * 70)
    
    for idx, path in enumerate(image_files, 1):
        print(f"\n[Image {idx}/3] Processing: {os.path.basename(path)}")
        img = cv2.imread(path)
        if img is None:
            print(f"  ✗ Could not read: {path}")
            continue

        # Quality check
        blur = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if blur < args.min_blur:
            print(f"  ✗ Skipping (too blurry {blur:.1f} < {args.min_blur}): {os.path.basename(path)}")
            continue

        # Detect with InsightFace
        faces = detector.get(img)
        if not faces:
            print(f"  ✗ No face detected: {os.path.basename(path)}")
            continue
        best = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)))
        kps = getattr(best, 'kps', None)
        if kps is None or len(kps) != 5:
            print(f"  ✗ Landmarks unavailable: {os.path.basename(path)}")
            continue

        # Align and embed original
        aligned = align_by_5pts(img, kps, out_size=(112, 112))
        if aligned is None:
            print(f"  ✗ Alignment failed: {os.path.basename(path)}")
            continue
        emb_orig = embed_aligned(aligned)
        if emb_orig is None:
            print(f"  ✗ Embedding failed: {os.path.basename(path)}")
            continue

        # Save original embedding
        orig_path = os.path.join(originals_dir, f"original_{idx}.npy")
        np.save(orig_path, emb_orig)
        saved_embeddings.append(orig_path)
        quality_scores.append(blur)
        metadata_images.append({
            'source': os.path.basename(path),
            'original_embedding': orig_path,
            'blur': float(blur)
        })
        print(f"  ✓ Original embedding saved: original_{idx}.npy (blur={blur:.1f})")

        # Generate masked variants (6 types)
        print(f"  • Generating {len(args.mask_types)} mask variations...")
        mask_save_root = os.path.join(masked_dir, f"img_{idx}")
        os.makedirs(mask_save_root, exist_ok=True)
        
        # Store masked images for visualization
        masked_images_for_viz = {}
        
        mask_success_count = 0
        for mtype in args.mask_types:
            # Use masktheface with proper config path
            masked_bgr = apply_mask_with_masktheface(img, mtype, dlib_detector, dlib_predictor)
            if masked_bgr is None:
                print(f"    - {mtype}: mask application failed")
                continue
            
            # Re-detect and align masked face
            faces_m = detector.get(masked_bgr)
            if not faces_m:
                print(f"    - {mtype}: no face detected after mask")
                continue
            best_m = max(faces_m, key=lambda f: float(getattr(f, 'det_score', 0.0)))
            kps_m = getattr(best_m, 'kps', None)
            if kps_m is None or len(kps_m) != 5:
                print(f"    - {mtype}: landmarks unavailable")
                continue
            aligned_m = align_by_5pts(masked_bgr, kps_m, out_size=(112, 112))
            if aligned_m is None:
                print(f"    - {mtype}: alignment failed")
                continue
            emb_m = embed_aligned(aligned_m)
            if emb_m is None:
                print(f"    - {mtype}: embedding failed")
                continue
            
            save_path = os.path.join(mask_save_root, f"{mtype}.npy")
            np.save(save_path, emb_m)
            saved_embeddings.append(save_path)
            masked_images_for_viz[mtype] = masked_bgr.copy()  # Store for visualization
            mask_success_count += 1
            print(f"    ✓ {mtype}: saved")

        # Create visualization for this image
        try:
            viz_path = os.path.join(visualizations_dir, f"image_{idx}_masks.jpg")
            create_mask_visualization(img, aligned, masked_images_for_viz, args.mask_types, viz_path)
            visualization_paths.append(viz_path)
            metadata_images[-1]['visualization'] = viz_path
            print(f"  ✓ Visualization saved: {os.path.basename(viz_path)}")
        except Exception as e:
            print(f"  ⚠️ Visualization failed: {e}")

        print(f"  → Image {idx}: 1 original + {mask_success_count} masked embeddings")

    if len(saved_embeddings) < 9:
        print(f"\n❌ Not enough embeddings saved ({len(saved_embeddings)}/27 expected).")
        print("   Ensure 3 good quality images with clear faces.")
        return 1

    # Compute centroid from originals only (quality-weighted)
    orig_paths = [os.path.join(originals_dir, f) for f in sorted(os.listdir(originals_dir)) if f.endswith('.npy')]
    orig_embs = [np.load(p) for p in orig_paths]
    if orig_embs:
        weights = np.array(quality_scores[:len(orig_embs)], dtype=np.float32)
        if weights.sum() <= 0:
            weights = np.ones(len(orig_embs), dtype=np.float32)
        weights = weights / (weights.sum() + 1e-12)
        stack = np.stack(orig_embs, axis=0)
        centroid = (weights[:, None] * stack).sum(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        np.save(os.path.join(user_dir, 'centroid.npy'), centroid.astype(np.float32))

        # Calculate recommended threshold
        sims = [cosine_similarity(e, centroid) for e in orig_embs]
        sim_mean = float(np.mean(sims))
        sim_std = float(np.std(sims))
        recommended_threshold = max(0.5, min(0.9, sim_mean - 2.0 * sim_std))
    else:
        recommended_threshold = 0.7

    # Create summary visualization if we have all visualizations
    summary_viz_path = None
    if len(visualization_paths) == 3:
        try:
            summary_viz_path = os.path.join(visualizations_dir, "summary_all_masks.jpg")
            create_summary_visualization(visualization_paths, summary_viz_path)
            print(f"\n✓ Summary visualization saved: {os.path.basename(summary_viz_path)}")
        except Exception as e:
            print(f"⚠️ Summary visualization failed: {e}")

    # Save metadata
    metadata = {
        'user_id': user_id,
        'user_name': user_name,
        'registration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'images_used': len(orig_embs),
        'total_embeddings': len(saved_embeddings),
        'mask_types': args.mask_types,
        'recommended_threshold': float(recommended_threshold),
        'structure': {
            'originals_dir': originals_dir,
            'masked_dir': masked_dir,
            'visualizations_dir': visualizations_dir
        },
        'quality_scores': [float(q) for q in quality_scores],
        'images': metadata_images,
        'visualizations': visualization_paths,
        'summary_visualization': summary_viz_path
    }
    with open(os.path.join(user_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*70)
    print("✅ REGISTRATION COMPLETE!")
    print("="*70)
    print(f"  • User: {user_name} (ID: {user_id})")
    print(f"  • Original embeddings: {len(orig_embs)}")
    print(f"  • Total embeddings (original + masked): {len(saved_embeddings)}")
    print(f"  • Recommended threshold: {recommended_threshold:.3f}")
    print(f"  • Directory: {user_dir}")
    print(f"\n📊 Visualizations saved in: {visualizations_dir}")
    print(f"   - Individual mask visualizations: {len(visualization_paths)} images")
    if summary_viz_path:
        print(f"   - Summary visualization: {os.path.basename(summary_viz_path)}")
    print("\nYou can now test recognition with:")
    print(f"  python recognize_mask_aug.py --cam 0")
    print("="*70)
    return 0

if __name__ == "__main__":
    code = main()
    sys.exit(code)

