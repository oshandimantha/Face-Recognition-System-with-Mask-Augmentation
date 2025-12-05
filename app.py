from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import cv2
import os
import sys
import json
from datetime import datetime
import numpy as np
import base64
import threading
import time
import atexit
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import registration and recognition classes (optional - handle if not available)
try:
    from src.core.registration import FaceRegistration
    from src.core.recognition import FaceRecognition
    REGISTRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Registration/Recognition modules not available: {e}")
    print("   Using standalone recognition system instead.")
    FaceRegistration = None
    FaceRecognition = None
    REGISTRATION_AVAILABLE = False

# Standalone registration system (fallback when src modules not available)
class StandaloneRegistration:
    """Standalone face registration using InsightFace, FaceNet, and mask augmentation"""
    def __init__(self):
        try:
            # Add masktheface to path if needed
            masktheface_path = os.path.join(PROJECT_ROOT, 'masktheface')
            if masktheface_path not in sys.path:
                sys.path.insert(0, masktheface_path)
            
            from insightface.app import FaceAnalysis
            from facenet_pytorch import InceptionResnetV1
            import torch
            import dlib
            from imutils import face_utils
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize detector
            print("🔄 Initializing InsightFace RetinaFace for registration...")
            self.detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.detector.prepare(ctx_id=0, det_size=(640, 640))
            
            # Initialize embedder
            print("🔄 Initializing VGGFace2 InceptionResnetV1 for registration...")
            self.model = InceptionResnetV1(pretrained='vggface2', classify=False).to(self.device).eval()
            
            # Initialize dlib for masktheface
            print("🔄 Initializing dlib for mask augmentation...")
            self.dlib_detector = dlib.get_frontal_face_detector()
            
            # Try multiple possible paths for dlib model
            dlib_model_path = None
            possible_paths = [
                os.path.join('dlib_models', 'shape_predictor_68_face_landmarks.dat'),
                os.path.join(PROJECT_ROOT, 'dlib_models', 'shape_predictor_68_face_landmarks.dat'),
                os.path.join('masktheface', 'dlib_models', 'shape_predictor_68_face_landmarks.dat'),
                os.path.join(PROJECT_ROOT, 'masktheface', 'dlib_models', 'shape_predictor_68_face_landmarks.dat'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    dlib_model_path = path
                    break
            
            if dlib_model_path and os.path.exists(dlib_model_path):
                try:
                    self.dlib_predictor = dlib.shape_predictor(dlib_model_path)
                    print(f"    ✓ dlib predictor loaded from {dlib_model_path}")
                except Exception as e:
                    print(f"    ⚠️ Failed to load dlib predictor: {e}")
                    self.dlib_predictor = None
            else:
                print("    ⚠️ dlib model not found at any of these paths:")
                for path in possible_paths:
                    print(f"       - {path}")
                print("    ⚠️ Mask augmentation will not work without dlib model")
                self.dlib_predictor = None
            
            print(f"✅ Standalone registration system initialized")
            
        except ImportError as e:
            print(f"❌ Failed to initialize standalone registration: {e}")
            print("   Please install: pip install insightface facenet-pytorch dlib imutils")
            self.detector = None
            self.model = None
            self.dlib_detector = None
            self.dlib_predictor = None
    
    def _align_face(self, image_bgr, kps5, out_size=(112, 112)):
        """Align face using 5-point landmarks"""
        ARCFACE_5PTS_112 = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        
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
    
    def _extract_embedding(self, aligned_rgb):
        """Extract embedding from aligned 112x112 RGB face"""
        if self.model is None:
            return None
        try:
            x = aligned_rgb.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]
            import torch
            t = torch.from_numpy(x).to(self.device)
            t = (t - 0.5) / 0.5
            with torch.no_grad():
                e = self.model(t)
                e = torch.nn.functional.normalize(e, p=2, dim=1)
            v = e.squeeze(0).cpu().numpy().astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            return v
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def _apply_mask(self, image_bgr, mask_type):
        """Apply mask using masktheface"""
        if self.dlib_predictor is None:
            print(f"    ⚠️ dlib predictor not available for mask {mask_type}")
            return None
        try:
            from masktheface.utils.aux_functions import (
                mask_face, get_six_points, shape_to_landmarks, rect_to_bb
            )
            from imutils import face_utils
            
            # Convert to RGB for masktheface
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect face with dlib
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            face_locations = self.dlib_detector(gray, 1)
            if not face_locations:
                print(f"    ⚠️ No face detected by dlib for mask {mask_type}")
                return None
            
            face_location = face_locations[0]
            shape = self.dlib_predictor(gray, face_location)
            shape = face_utils.shape_to_np(shape)
            face_landmarks = shape_to_landmarks(shape)
            face_location_bb = rect_to_bb(face_location)
            
            # Get six points and angle
            six_points, angle = get_six_points(face_landmarks, image_rgb)
            
            # Create args object
            class Args:
                def __init__(self):
                    self.pattern = ""
                    self.pattern_weight = 0.5
                    self.color = ""
                    self.color_weight = 0.5
                    self.verbose = False
            
            args = Args()
            
            # Change to project root for masktheface config
            original_dir = os.getcwd()
            try:
                os.chdir(PROJECT_ROOT)
                masked_image = mask_face(
                    image_rgb, face_location_bb, six_points, angle, args, 
                    type=mask_type, return_mask_status=False
                )
            finally:
                os.chdir(original_dir)
            
            if masked_image is None:
                print(f"    ⚠️ mask_face returned None for {mask_type}")
                return None
            
            return cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        except ImportError as e:
            print(f"    ❌ Import error applying mask {mask_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"    ❌ Error applying mask {mask_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def register_user(self, user_id, user_name, mobile_number, image_paths, camera_index=0):
        """Register a user with mask augmentation"""
        if self.detector is None or self.model is None:
            return False
        
        try:
            # Create user directory
            user_dir = os.path.join('embeddings', f"{user_id}_{user_name}")
            os.makedirs(user_dir, exist_ok=True)
            
            originals_dir = os.path.join(user_dir, 'originals')
            masked_dir = os.path.join(user_dir, 'masked')
            visualizations_dir = os.path.join(user_dir, 'visualizations')
            os.makedirs(originals_dir, exist_ok=True)
            os.makedirs(masked_dir, exist_ok=True)
            os.makedirs(visualizations_dir, exist_ok=True)
            
            # Mask types to apply
            mask_types = ['surgical_blue', 'surgical_green', 'cloth', 'n95', 'kn95', 'gas']
            
            saved_embeddings = []
            quality_scores = []
            metadata_images = []
            
            print(f"📁 Processing {len(image_paths)} images with mask augmentation...")
            
            for idx, img_path in enumerate(image_paths, 1):
                print(f"  [Image {idx}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
                img = cv2.imread(img_path)
                if img is None:
                    print(f"    ✗ Could not read image")
                    continue
                
                # Quality check (blur)
                blur = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if blur < 40.0:
                    print(f"    ✗ Too blurry (blur={blur:.1f})")
                    continue
                
                # Detect face
                faces = self.detector.get(img)
                if not faces:
                    print(f"    ✗ No face detected")
                    continue
                
                best = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)))
                kps = getattr(best, 'kps', None)
                if kps is None or len(kps) != 5:
                    print(f"    ✗ Landmarks unavailable")
                    continue
                
                # Align and embed original
                aligned = self._align_face(img, kps, out_size=(112, 112))
                if aligned is None:
                    print(f"    ✗ Alignment failed")
                    continue
                
                emb_orig = self._extract_embedding(aligned)
                if emb_orig is None:
                    print(f"    ✗ Embedding failed")
                    continue
                
                # Save original embedding
                orig_path = os.path.join(originals_dir, f"original_{idx}.npy")
                np.save(orig_path, emb_orig)
                saved_embeddings.append(emb_orig)
                quality_scores.append(blur)
                metadata_images.append({
                    'source': os.path.basename(img_path),
                    'original_embedding': f"original_{idx}.npy",
                    'blur': float(blur)
                })
                print(f"    ✓ Original embedding saved (blur={blur:.1f})")
                
                # Generate masked variants
                mask_save_root = os.path.join(masked_dir, f"img_{idx}")
                os.makedirs(mask_save_root, exist_ok=True)
                
                mask_success_count = 0
                for mtype in mask_types:
                    print(f"      • Applying {mtype}...")
                    masked_bgr = self._apply_mask(img, mtype)
                    if masked_bgr is None:
                        print(f"      ✗ {mtype}: mask application failed")
                        continue
                    
                    # Re-detect and align masked face
                    faces_m = self.detector.get(masked_bgr)
                    if not faces_m:
                        print(f"      ✗ {mtype}: no face detected after mask")
                        continue
                    
                    best_m = max(faces_m, key=lambda f: float(getattr(f, 'det_score', 0.0)))
                    kps_m = getattr(best_m, 'kps', None)
                    if kps_m is None or len(kps_m) != 5:
                        print(f"      ✗ {mtype}: landmarks unavailable")
                        continue
                    
                    aligned_m = self._align_face(masked_bgr, kps_m, out_size=(112, 112))
                    if aligned_m is None:
                        print(f"      ✗ {mtype}: alignment failed")
                        continue
                    
                    emb_m = self._extract_embedding(aligned_m)
                    if emb_m is None:
                        print(f"      ✗ {mtype}: embedding extraction failed")
                        continue
                    
                    # Save masked embedding
                    mask_path = os.path.join(mask_save_root, f"{mtype}.npy")
                    np.save(mask_path, emb_m)
                    saved_embeddings.append(emb_m)
                    mask_success_count += 1
                    print(f"      ✓ {mtype}: saved")
                
                print(f"    ✓ Generated {mask_success_count}/{len(mask_types)} masked embeddings")
            
            if not saved_embeddings:
                print("❌ No embeddings saved")
                return False
            
            # Calculate centroid
            centroid = np.mean(saved_embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            centroid_path = os.path.join(user_dir, 'centroid.npy')
            np.save(centroid_path, centroid)
            
            # Save metadata
            metadata = {
                'user_id': user_id,
                'user_name': user_name,
                'mobile_number': mobile_number,
                'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_embeddings': len(saved_embeddings),
                'original_images': len(image_paths),
                'quality_scores': quality_scores,
                'images': metadata_images,
                'recommended_threshold': 0.7
            }
            metadata_path = os.path.join(user_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Registration successful!")
            print(f"   Saved {len(saved_embeddings)} embeddings")
            print(f"   User directory: {user_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Registration error: {e}")
            import traceback
            traceback.print_exc()
            return False

# Standalone recognition system (fallback when src modules not available)
class StandaloneRecognition:
    """Standalone face recognition using InsightFace and FaceNet"""
    def __init__(self):
        try:
            from insightface.app import FaceAnalysis
            from facenet_pytorch import InceptionResnetV1
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize detector
            print("🔄 Initializing InsightFace RetinaFace...")
            self.detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.detector.prepare(ctx_id=0, det_size=(640, 640))
            
            # Initialize embedder
            print("🔄 Initializing VGGFace2 InceptionResnetV1...")
            self.model = InceptionResnetV1(pretrained='vggface2', classify=False).to(self.device).eval()
            
            # Load embeddings
            self.known_embeddings = {}
            self.user_info = {}
            self._load_embeddings()
            
            print(f"✅ Standalone recognition system initialized")
            print(f"   Loaded {len(self.known_embeddings)} registered user(s)")
            
        except ImportError as e:
            print(f"❌ Failed to initialize standalone recognition: {e}")
            print("   Please install: pip install insightface facenet-pytorch")
            self.detector = None
            self.model = None
            self.known_embeddings = {}
            self.user_info = {}
    
    def _load_embeddings(self):
        """Load embeddings from embeddings directory"""
        import json
        import glob
        
        if not os.path.exists('embeddings'):
            return
        
        for user_dir in os.listdir('embeddings'):
            user_path = os.path.join('embeddings', user_dir)
            if not os.path.isdir(user_path):
                continue
            
            # Parse user info from directory name
            if '_' in user_dir:
                user_id, user_name = user_dir.split('_', 1)
            else:
                user_id = user_dir
                user_name = user_dir
            
            # Skip test entries
            if user_id.lower() == 'img' or (user_name.isdigit() and len(user_name) <= 2):
                continue
            
            # Load all embeddings (original + masked) for better accuracy
            embeddings_list = []
            
            # Load originals
            originals_dir = os.path.join(user_path, 'originals')
            if os.path.exists(originals_dir):
                for f in sorted(os.listdir(originals_dir)):
                    if f.endswith('.npy'):
                        try:
                            emb = np.load(os.path.join(originals_dir, f))
                            emb = emb / (np.linalg.norm(emb) + 1e-12)
                            embeddings_list.append(emb)
                        except Exception as e:
                            print(f"⚠️ Error loading original embedding {f}: {e}")
            
            # Load masked embeddings
            masked_dir = os.path.join(user_path, 'masked')
            if os.path.exists(masked_dir):
                for img_sub in sorted(os.listdir(masked_dir)):
                    sub = os.path.join(masked_dir, img_sub)
                    if not os.path.isdir(sub):
                        continue
                    for f in sorted(os.listdir(sub)):
                        if f.endswith('.npy'):
                            try:
                                emb = np.load(os.path.join(sub, f))
                                emb = emb / (np.linalg.norm(emb) + 1e-12)
                                embeddings_list.append(emb)
                            except Exception as e:
                                pass  # Silently skip errors
            
            # If no embeddings found, try centroid as fallback
            if not embeddings_list:
                centroid_path = os.path.join(user_path, 'centroid.npy')
                if os.path.exists(centroid_path):
                    try:
                        embedding = np.load(centroid_path)
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
                        embeddings_list.append(embedding)
                    except Exception as e:
                        print(f"⚠️ Error loading centroid for {user_name}: {e}")
            
            if embeddings_list:
                # Store all embeddings for this user
                self.known_embeddings[user_name] = embeddings_list
                
                # Load metadata
                metadata_path = os.path.join(user_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            self.user_info[user_name] = json.load(f)
                    except Exception as e:
                        print(f"⚠️ Error loading metadata for {user_name}: {e}")
                print(f"  ✓ Loaded {len(embeddings_list)} embeddings for {user_name}")
    
    def _align_face(self, image_bgr, kps5, out_size=(112, 112)):
        """Align face using 5-point landmarks"""
        ARCFACE_5PTS_112 = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        
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
    
    def _extract_embedding(self, image_bgr):
        """Extract face embedding from BGR image"""
        if self.detector is None or self.model is None:
            return None, None
        
        try:
            faces = self.detector.get(image_bgr)
            if not faces:
                return None, None
            
            best = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)))
            kps = getattr(best, 'kps', None)
            if kps is None or len(kps) != 5:
                return None, None
            
            aligned = self._align_face(image_bgr, kps, out_size=(112, 112))
            if aligned is None:
                return None, None
            
            # Get bounding box
            bbox_attr = getattr(best, 'bbox', None)
            bbox = None
            if bbox_attr is not None:
                # bbox format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = int(bbox_attr[0]), int(bbox_attr[1]), int(bbox_attr[2]), int(bbox_attr[3])
                bbox = [x1, y1, x2-x1, y2-y1]
            
            # Extract embedding
            x = aligned.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]
            import torch
            t = torch.from_numpy(x).to(self.device)
            t = (t - 0.5) / 0.5
            
            with torch.no_grad():
                e = self.model(t)
                e = torch.nn.functional.normalize(e, p=2, dim=1)
            
            v = e.squeeze(0).cpu().numpy().astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            
            return v, bbox
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def recognize_image(self, image_bgr, debug=False):
        """Recognize face in image - returns (user_name, similarity, status_message)"""
        if len(self.known_embeddings) == 0:
            return None, 0.0, "No registered users"
        
        embedding, bbox = self._extract_embedding(image_bgr)
        if embedding is None:
            return None, 0.0, "No face detected"
        
        # Find best match (compare against all embeddings for each user)
        best_name = None
        best_score = -1.0
        threshold = 0.65
        
        for user_name, known_embs in self.known_embeddings.items():
            # known_embs can be a list of embeddings or a single embedding
            if isinstance(known_embs, list):
                # Compare against all embeddings and take max similarity
                scores = [float(np.dot(embedding, emb)) for emb in known_embs]
                score = max(scores) if scores else -1.0
            else:
                # Single embedding
                score = float(np.dot(embedding, known_embs))
            
            if score > best_score:
                best_score = score
                best_name = user_name
        
        if best_score >= threshold:
            return best_name, best_score, "Recognized"
        else:
            return None, best_score, f"Below threshold ({best_score:.3f} < {threshold:.3f})"
    
    def detect_and_align(self, image_bgr):
        """Detect and align face - returns (aligned_face, bbox) for compatibility"""
        embedding, bbox = self._extract_embedding(image_bgr)
        if embedding is None:
            return None, None
        
        # Return a dummy aligned face (we already have the embedding)
        # This is for compatibility with the existing code
        return np.zeros((112, 112, 3), dtype=np.uint8), bbox

app = Flask(__name__, static_folder='static', template_folder='templates')

# Performance tracking
performance_stats = {
    'total_recognitions': 0,
    'successful_recognitions': 0,
    'failed_recognitions': 0,
    'average_confidence': 0.0,
    'recognition_times': []
}

# Initialize the registration and recognition classes
registration_system = None
recognition_system = None

if REGISTRATION_AVAILABLE:
    try:
        print("🔄 Initializing recognition systems...")
        start_time = time.time()
        registration_system = FaceRegistration()
        recognition_system = FaceRecognition()
        init_time = time.time() - start_time
        print(f"✅ Recognition systems initialized successfully in {init_time:.2f}s")
        print(f"📊 System Configuration:")
        print(f"   - Model: Face Recognition")
        print(f"   - Registered Users: {len(recognition_system.known_embeddings)}")
    except Exception as e:
        print(f"❌ Error initializing recognition systems: {e}")
        import traceback
        traceback.print_exc()
        registration_system = None
        recognition_system = None
        # Fall back to standalone systems
        print("🔄 Falling back to standalone systems...")
        registration_system = StandaloneRegistration()
        recognition_system = StandaloneRecognition()
else:
    print("⚠️ Registration/Recognition modules not available")
    print("🔄 Using standalone systems...")
    registration_system = StandaloneRegistration()
    recognition_system = StandaloneRecognition()

# Backend wrapper functions
def register_user_backend(name, worker_id, mobile, image_paths):
    """Wrapper function for registration backend"""
    try:
        # Check if registration system is available
        if registration_system is None:
            return False, "Registration system not initialized"
        
        print(f"🔄 Starting registration for {name} (ID: {worker_id})")
        print(f"📁 Processing {len(image_paths)} images")
        
        # Check if it's StandaloneRegistration or FaceRegistration
        if isinstance(registration_system, StandaloneRegistration):
            # Use standalone registration
            success = registration_system.register_user(
                user_id=worker_id,
                user_name=name,
                mobile_number=mobile,
                image_paths=image_paths,
                camera_index=0  # Not used when image_paths provided
            )
        else:
            # Use FaceRegistration.register_user() method
            success = registration_system.register_user(
                user_id=worker_id,
                user_name=name,
                mobile_number=mobile,
                image_paths=image_paths,
                camera_index=0  # Not used when image_paths provided
            )
        
        if success:
            # Reload recognition system to include new user
            if isinstance(recognition_system, StandaloneRecognition):
                recognition_system._load_embeddings()
            return True, f"Registration successful! Processed {len(image_paths)} images."
        else:
            return False, f"Registration failed. Please ensure all images contain clear faces."
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Registration error: {str(e)}"

def recognize_user_backend(frame):
    """Wrapper function for recognition backend"""
    try:
        # Check if recognition system is available
        if recognition_system is None:
            return "Unknown", "System Error", "---", "Recognition system not initialized", None
        
        # Check if it's StandaloneRecognition or FaceRecognition
        if isinstance(recognition_system, StandaloneRecognition):
            # Use standalone recognition - extract embedding once and get bbox
            embedding, bbox = recognition_system._extract_embedding(frame)
            if embedding is None:
                return "Unknown", "No Face", "---", "No face detected", None
            
            # Find best match (compare against all embeddings for each user)
            best_name = None
            best_score = -1.0
            threshold = 0.65
            
            for user_name, known_embs in recognition_system.known_embeddings.items():
                # known_embs can be a list of embeddings or a single embedding
                if isinstance(known_embs, list):
                    # Compare against all embeddings and take max similarity
                    scores = [float(np.dot(embedding, emb)) for emb in known_embs]
                    score = max(scores) if scores else -1.0
                else:
                    # Single embedding
                    score = float(np.dot(embedding, known_embs))
                
                if score > best_score:
                    best_score = score
                    best_name = user_name
            
            user_name = best_name if best_score >= threshold else None
            similarity = best_score
            status_message = "Recognized" if user_name else f"Below threshold ({best_score:.3f} < {threshold:.3f})"
        else:
            # Use FaceRecognition.recognize_image() method
            user_name, similarity, status_message = recognition_system.recognize_image(frame, debug=False)
            # Get bounding box from face aligner if available
            bbox = None
            if hasattr(recognition_system, 'face_aligner'):
                aligned_face, box = recognition_system.face_aligner.detect_and_align(frame)
                if box is not None:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    bbox = [x1, y1, x2-x1, y2-y1]
        
        # Track performance
        performance_stats['total_recognitions'] += 1
        
        # Convert similarity to confidence (similarity is 0-1, higher is better)
        confidence = similarity
        
        # Get threshold from settings if available, otherwise use default
        try:
            import config.settings as settings
            threshold = settings.RECOGNITION_CONFIG.get('min_similarity_threshold', 0.65)
        except:
            threshold = 0.65
        
        if user_name is not None and confidence >= threshold:
            # Get user info if available
            if hasattr(recognition_system, 'user_info') and user_name in recognition_system.user_info:
                user_info = recognition_system.user_info[user_name]
                user_id = user_info.get('user_id', user_name)
                display_name = user_info.get('user_name', user_name)
            else:
                # Try to extract from directory name
                if '_' in user_name:
                    parts = user_name.split('_', 1)
                    user_id = parts[0] if parts[0].isdigit() else user_name
                    display_name = parts[1] if len(parts) > 1 else user_name
                else:
                    user_id = user_name
                    display_name = user_name
            
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Track successful recognition
            performance_stats['successful_recognitions'] += 1
            performance_stats['average_confidence'] = (
                (performance_stats['average_confidence'] * (performance_stats['successful_recognitions'] - 1) + confidence) 
                / performance_stats['successful_recognitions']
            )
            
            print(f"✅ Recognition Success: {display_name} (similarity: {confidence:.3f})")
            return display_name, "Present", current_time, f"Recognized with similarity {confidence:.3f}", bbox
        else:
            # Person detected but not in database
            performance_stats['failed_recognitions'] += 1
            print(f"❌ Recognition Failed: {user_name or 'Unknown'} (similarity: {confidence:.3f})")

            return "Unknown", "Not in Database", "---", f"Person not found in database - similarity: {confidence:.3f}", None
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "Unknown", "Error", "---", f"Recognition error: {str(e)}", None

def check_photo_quality(image_path):
    """Check if uploaded photo meets quality requirements"""
    try:
        if registration_system is None:
            return False
        
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Check if it's StandaloneRegistration or FaceRegistration
        if isinstance(registration_system, StandaloneRegistration):
            # Use standalone detection
            if registration_system.detector is None:
                return False
            
            # Check blur quality
            blur = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if blur < 40.0:
                return False
            
            # Check if face is detected
            faces = registration_system.detector.get(img)
            if not faces:
                return False
            
            return True
        else:
            # Use FaceRegistration's face aligner to detect and align
            aligned_face, _ = registration_system.face_aligner.detect_and_align(img)
            if aligned_face is None:
                return False
            
            # Use quality checker
            is_good, msg, score = registration_system.quality_checker.check_quality(aligned_face)
            return is_good
        
    except Exception as e:
        return False

# Global camera instance for better resource management
camera = None
camera_lock = threading.Lock()

def init_camera():
    """Initialize camera with better error handling and fallback options"""
    global camera
    with camera_lock:
        if camera is not None:
            return camera
            
        # Force use of camera index 1 (USB camera)
        camera_options = [
            (1, cv2.CAP_DSHOW),  # USB camera (index 1) with DirectShow - PRIMARY
            (1, cv2.CAP_ANY),    # USB camera (index 1) with any backend - FALLBACK
            (0, cv2.CAP_DSHOW),  # Laptop camera as last resort
            (0, cv2.CAP_ANY),    # Laptop camera with any backend as last resort
        ]
        
        for camera_index, backend in camera_options:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    # Test if camera actually works
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Set optimal camera properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus to reduce conflicts
                        camera = cap
                        print(f"Camera initialized successfully with index {camera_index}")
                        return camera
                    else:
                        cap.release()
                        print(f"Camera {camera_index} opened but failed to read frame")
            except Exception as e:
                print(f"Failed to initialize camera {camera_index}: {e}")
                # If camera is in use, try to release it and wait
                if "in use" in str(e).lower() or "busy" in str(e).lower():
                    print(f"Camera {camera_index} is in use, trying to release...")
                    try:
                        cap.release()
                        time.sleep(1)  # Wait 1 second before trying again
                    except:
                        pass
                continue
        
        print("No working camera found")
        return None

def cleanup_camera():
    """Clean up camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
                print("Camera released successfully")
            except Exception as e:
                print(f"Error releasing camera: {e}")
            finally:
                camera = None

def force_release_all_cameras():
    """Force release all camera resources with improved conflict resolution"""
    global camera
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
                print("Released main camera")
            except:
                pass
            camera = None
    
    # Force release cameras with smart detection
    # Only try indices that are likely to exist based on system
    camera_indices_to_try = [1, 0]  # Try USB camera first, then laptop camera
    
    for i in camera_indices_to_try:
        for attempt in range(2):  # Try 2 times per camera
            try:
                # Use DirectShow backend to avoid index conflicts
                temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if temp_cap.isOpened():
                    temp_cap.release()
                    print(f"Released camera at index {i}")
                    break
                else:
                    # Camera not available at this index, no error
                    break
            except Exception as e:
                # Silently handle camera index errors - this is normal
                if attempt == 1:  # Last attempt
                    print(f"Camera {i} not available (normal)")
                time.sleep(0.1)  # Brief pause between attempts
    
    # Additional cleanup - try to release any remaining camera resources
    try:
        cv2.destroyAllWindows()
        print("Cleaned up OpenCV windows")
    except:
        pass

# Webcam stream generator with improved error handling
def gen_frames():
    global camera
    
    # Initialize camera if not already done
    if camera is None:
        camera = init_camera()
        if camera is None:
            # Send error frame with more helpful message
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Not Available", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_frame, "Check if camera is being used", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(error_frame, "by another application", (50, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
    
    frame_count = 0
    consecutive_failures = 0
    max_failures = 10
    
    while True:
        try:
            ret, frame = camera.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print("Too many consecutive camera failures, reinitializing...")
                    cleanup_camera()
                    camera = init_camera()
                    consecutive_failures = 0
                    if camera is None:
                        break
                continue
            
            consecutive_failures = 0
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add frame counter for debugging
            if frame_count % 100 == 0:
                print(f"Streaming frame {frame_count}")
            
            # Encode frame with optimized settings
            ret, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 85,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print("Failed to encode frame")
                
        except Exception as e:
            print(f"Error in video stream: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                break
            time.sleep(0.1)  # Brief pause before retry

# Serve HTML pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/user')
def user():
    return render_template('user.html')

# Webcam video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to force release cameras
@app.route('/release_cameras', methods=['POST'])
def release_cameras():
    try:
        force_release_all_cameras()
        return jsonify({'success': True, 'message': 'Cameras released successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error releasing cameras: {str(e)}'})

# Route to check camera status
@app.route('/camera_status', methods=['GET'])
def camera_status():
    try:
        # Check if camera is available - try USB camera first, then laptop camera
        camera_indices = [1, 0]  # USB camera first, then laptop camera
        
        for camera_index in camera_indices:
            try:
                test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    test_cap.release()
                    if ret and frame is not None:
                        return jsonify({
                            'available': True, 
                            'message': f'Camera is available (index {camera_index})',
                            'camera_index': camera_index
                        })
                    else:
                        continue  # Try next camera index
                else:
                    continue  # Try next camera index
            except Exception as e:
                continue  # Try next camera index
        
        # If we get here, no cameras are available
        return jsonify({'available': False, 'message': 'No cameras are available or all are in use'})
        
    except Exception as e:
        return jsonify({'available': False, 'message': f'Camera error: {str(e)}'})

# Route to detect face without full recognition
@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        if 'image' not in request.files:
            return jsonify({'face_detected': False, 'message': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'face_detected': False, 'message': 'No image selected'})
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'face_detected': False, 'message': 'Invalid image'})
        
        # Check if recognition system is properly initialized
        if recognition_system is None:
            return jsonify({'face_detected': False, 'message': 'Recognition system not initialized'})
        
        # Use appropriate detection method based on recognition system type
        if isinstance(recognition_system, StandaloneRecognition):
            # Use InsightFace detector
            if recognition_system.detector is None:
                return jsonify({'face_detected': False, 'message': 'Detector not initialized'})
            
            faces = recognition_system.detector.get(img)
            if faces and len(faces) > 0:
                best = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)))
                bbox_attr = getattr(best, 'bbox', None)
                if bbox_attr is not None:
                    x1, y1, x2, y2 = int(bbox_attr[0]), int(bbox_attr[1]), int(bbox_attr[2]), int(bbox_attr[3])
                    bbox = [x1, y1, x2-x1, y2-y1]
                    return jsonify({
                        'face_detected': True, 
                        'bbox': bbox,
                        'message': f'Face detected'
                    })
            return jsonify({'face_detected': False, 'message': 'No face detected'})
        else:
            # Use FaceRecognition's face aligner to detect
            if hasattr(recognition_system, 'face_aligner'):
                aligned_face, box = recognition_system.face_aligner.detect_and_align(img)
                if aligned_face is not None and box is not None:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    bbox = [x1, y1, x2-x1, y2-y1]
                    return jsonify({
                        'face_detected': True, 
                        'bbox': bbox,
                        'message': f'Face detected'
                    })
            return jsonify({'face_detected': False, 'message': 'No face detected'})
            
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({'face_detected': False, 'message': f'Detection error: {str(e)}'})

# Route to log access attempts
@app.route('/log_access', methods=['POST'])
def log_access():
    try:
        data = request.get_json()
        name = data.get('name', 'Unknown')
        status = data.get('status', 'denied')
        time = data.get('time', datetime.now().strftime("%H:%M:%S"))
        reason = data.get('reason', 'No reason provided')
        
        # Create access log entry with proper date/time
        current_datetime = datetime.now()
        log_entry = {
            'date': current_datetime.strftime("%Y-%m-%d"),
            'time': current_datetime.strftime("%H:%M:%S"),
            'name': name,
            'status': status,
            'reason': reason
        }
        
        # Save to access log file
        access_log_file = 'user_access_log.csv'
        file_exists = os.path.exists(access_log_file)
        
        with open(access_log_file, 'a', newline='', encoding='utf-8') as f:
            if not file_exists:
                f.write("Date,Time,Name,Status,Reason\n")
            f.write(f"{log_entry['date']},{log_entry['time']},{log_entry['name']},{log_entry['status']},{log_entry['reason']}\n")
        
        print(f"🔐 Access logged: {name} - {status} at {log_entry['time']}")
        return jsonify({'success': True, 'message': 'Access logged successfully'})
        
    except Exception as e:
        print(f"❌ Error logging access: {e}")
        return jsonify({'success': False, 'message': f'Error logging access: {str(e)}'})

# Route to get user statistics
@app.route('/user_stats', methods=['GET'])
def get_user_stats():
    try:
        # Count registered users (only top-level user directories, excluding test entries)
        registered_count = 0
        if os.path.exists('embeddings'):
            for user_dir in os.listdir('embeddings'):
                user_path = os.path.join('embeddings', user_dir)
                if not os.path.isdir(user_path):
                    continue
                
                # Parse user info from directory name
                if '_' in user_dir:
                    user_id, user_name = user_dir.split('_', 1)
                else:
                    user_id = user_dir
                    user_name = user_dir
                
                # Filter out test entries (same logic as /registered_users)
                if user_id.lower() == 'img':
                    continue
                if user_name.isdigit() and len(user_name) <= 2:
                    continue
                if user_id.isdigit() and user_name.isdigit() and len(user_name) <= 2:
                    continue
                
                # Check if this directory has embeddings (centroid or originals)
                centroid_path = os.path.join(user_path, 'centroid.npy')
                originals_dir = os.path.join(user_path, 'originals')
                has_embeddings = os.path.exists(centroid_path) or (os.path.exists(originals_dir) and any(f.endswith('.npy') for f in os.listdir(originals_dir) if os.path.isfile(os.path.join(originals_dir, f))))
                
                if has_embeddings:
                    registered_count += 1
        
        # Count today's access attempts
        today_access = 0
        total_access = 0
        today = datetime.now().strftime("%Y-%m-%d")
        access_log_file = 'user_access_log.csv'
        
        if os.path.exists(access_log_file):
            with open(access_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    total_access += 1
                    if today in line:
                        today_access += 1
        
        return jsonify({
            'success': True,
            'registered_users': registered_count,
            'today_access': today_access,
            'total_access': total_access
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting stats: {str(e)}'})

# Route to get all registered users
@app.route('/registered_users', methods=['GET'])
def get_registered_users():
    try:
        import json
        users = []
        if os.path.exists('embeddings'):
            for root, dirs, files in os.walk('embeddings'):
                if files and any(f.endswith('.npy') for f in files):
                    # Extract user info from directory name
                    dir_name = os.path.basename(root)
                    if '_' in dir_name:
                        user_id, user_name = dir_name.split('_', 1)
                        
                        # Filter out test/invalid entries:
                        # - Skip if user_id is "img" (test entries from image uploads)
                        # - Skip if user_name is just a number (like "1", "2", "3")
                        # - Skip if user_id is just a number and user_name is also just a number
                        if user_id.lower() == 'img':
                            continue
                        if user_name.isdigit() and len(user_name) <= 2:
                            continue
                        if user_id.isdigit() and user_name.isdigit() and len(user_name) <= 2:
                            continue
                        
                        # Load metadata.json if it exists to get mobile number
                        mobile_number = 'N/A'
                        metadata_path = os.path.join(root, 'metadata.json')
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    metadata = json.load(f)
                                    # Try different possible field names for mobile number
                                    mobile_number = metadata.get('mobile_number') or \
                                                   metadata.get('mobile') or \
                                                   metadata.get('phone') or \
                                                   metadata.get('phone_number') or 'N/A'
                            except Exception as e:
                                print(f"Error loading metadata for {user_name}: {e}")
                        
                        users.append({
                            'id': user_id,
                            'name': user_name,
                            'mobile': mobile_number if mobile_number != 'N/A' else 'Not provided',
                            'directory': root
                        })
        
        return jsonify({'success': True, 'users': users})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting users: {str(e)}'})

# Route to get access log
@app.route('/access_log', methods=['GET'])
def get_access_log():
    try:
        access_log_file = 'user_access_log.csv'
        logs = []
        
        if os.path.exists(access_log_file):
            with open(access_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        logs.append({
                            'date': parts[0],
                            'time': parts[1],
                            'name': parts[2],
                            'status': parts[3],
                            'reason': parts[4]
                        })
        
        # Sort by date and time (newest first)
        logs.sort(key=lambda x: f"{x['date']} {x['time']}", reverse=True)
        
        return jsonify({'success': True, 'logs': logs})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting access log: {str(e)}'})

# Route to export access log
@app.route('/export_access_log', methods=['GET'])
def export_access_log():
    try:
        access_log_file = 'user_access_log.csv'
        
        if not os.path.exists(access_log_file):
            return jsonify({'success': False, 'message': 'No access log found'})
        
        with open(access_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Response(
            content,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=access_log_{datetime.now().strftime("%Y%m%d")}.csv'}
        )
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error exporting access log: {str(e)}'})

# Route to clear access log
@app.route('/clear_access_log', methods=['POST'])
def clear_access_log():
    try:
        access_log_file = 'user_access_log.csv'
        
        if os.path.exists(access_log_file):
            # Create backup before clearing
            backup_file = f'access_log_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            import shutil
            shutil.copy2(access_log_file, backup_file)
            
            # Clear the log file but keep header
            with open(access_log_file, 'w', encoding='utf-8') as f:
                f.write("Date,Time,Name,Status,Reason\n")
        
        print(f"🗑️ Access log cleared (backup: {backup_file})")
        return jsonify({'success': True, 'message': 'Access log cleared successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error clearing access log: {str(e)}'})

# Route to delete user embeddings
@app.route('/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        user_name = data.get('user_name')
        
        if not user_id or not user_name:
            return jsonify({'success': False, 'message': 'User ID and name required'})
        
        # Find user directory
        user_dir = os.path.join('embeddings', f"{user_id}_{user_name}")
        
        if not os.path.exists(user_dir):
            return jsonify({'success': False, 'message': 'User not found'})
        
        # Delete user directory and all embeddings
        import shutil
        shutil.rmtree(user_dir)
        
        # Also delete debug images if exists
        debug_dir = os.path.join('debug_images', f"{user_id}_{user_name}")
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        
        print(f"🗑️ Deleted user: {user_name} (ID: {user_id})")
        return jsonify({'success': True, 'message': f'User {user_name} deleted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error deleting user: {str(e)}'})

# Route to serve user access management page
@app.route('/user_access')
def user_access():
    return render_template('user_access.html')

# Registration endpoint: accept name, worker_id, mobile, 3 images
@app.route('/register', methods=['POST'])
def register():
    try:
        # Validate required fields
        name = request.form.get('name', '').strip()
        worker_id = request.form.get('worker_id', '').strip()
        mobile = request.form.get('mobile', '').strip()
        
        if not name or not worker_id or not mobile:
            return jsonify({'success': False, 'message': 'All fields (name, worker_id, mobile) are required'})
        
        # Validate worker_id format (alphanumeric)
        if not worker_id.replace('_', '').replace('-', '').isalnum():
            return jsonify({'success': False, 'message': 'Worker ID must contain only letters, numbers, hyphens, and underscores'})
        
        # Validate mobile number (basic check)
        if not mobile.replace('+', '').replace('-', '').replace(' ', '').isdigit() or len(mobile.replace('+', '').replace('-', '').replace(' ', '')) < 7:
            return jsonify({'success': False, 'message': 'Please enter a valid mobile number'})
        
        # Get uploaded images
        images = [
            request.files.get('image1'),
            request.files.get('image2'),
            request.files.get('image3')
        ]
        
        if not all(images):
            return jsonify({'success': False, 'message': 'All 3 images are required'})
        
        # Validate image files
        for idx, img in enumerate(images):
            if not img or img.filename == '':
                return jsonify({'success': False, 'message': f'Image {idx+1} is required'})
            
            # Check file size (max 10MB)
            img.seek(0, 2)  # Seek to end
            file_size = img.tell()
            img.seek(0)  # Reset to beginning
            if file_size > 10 * 1024 * 1024:  # 10MB
                return jsonify({'success': False, 'message': f'Image {idx+1} is too large (max 10MB)'})
            
            # Check file type
            if not img.content_type or not img.content_type.startswith('image/'):
                return jsonify({'success': False, 'message': f'Image {idx+1} must be a valid image file'})
        
        # Create user directory
        user_dir = os.path.join('temp_uploads', f"{worker_id}_{name}")
        try:
            os.makedirs(user_dir, exist_ok=True)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Failed to create user directory: {str(e)}'})
        
        # Save images with error handling
        image_paths = []
        for idx, img in enumerate(images):
            try:
                img_path = os.path.join(user_dir, f"upload_{idx+1}.jpg")
                img.save(img_path)
                image_paths.append(img_path)
            except Exception as e:
                return jsonify({'success': False, 'message': f'Failed to save image {idx+1}: {str(e)}'})
        
        # Process registration
        success, message = register_user_backend(name, worker_id, mobile, image_paths)
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'})

# Recognition endpoint
@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        img_file = request.files.get('image')
        if not img_file or img_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No image uploaded',
                'name': '---',
                'status': '---',
                'time': '---'
            })
        
        # Validate file size
        img_file.seek(0, 2)  # Seek to end
        file_size = img_file.tell()
        img_file.seek(0)  # Reset to beginning
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({
                'success': False,
                'message': 'Image too large (max 10MB)',
                'name': '---',
                'status': '---',
                'time': '---'
            })
        
        # Read image as numpy array
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                'success': False,
                'message': 'Invalid image data - could not decode image',
                'name': '---',
                'status': '---',
                'time': '---'
            })
        
        # Check image dimensions
        if frame.shape[0] < 50 or frame.shape[1] < 50:
            return jsonify({
                'success': False,
                'message': 'Image too small for face detection',
                'name': '---',
                'status': '---',
                'time': '---'
            })
        
        # Call recognition logic
        name, status, time_str, message, bbox = recognize_user_backend(frame)
        
        return jsonify({
            'success': True,
            'message': message,
            'name': name,
            'status': status,
            'time': time_str,
            'bbox': bbox
        })
        
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({
            'success': False,
            'message': f'Recognition failed: {str(e)}',
            'name': '---',
            'status': '---',
            'time': '---'
        })

@app.route('/check_photo', methods=['POST'])
def check_photo():
    try:
        img = request.files.get('image')
        if not img:
            return jsonify({'qualified': False})
        # Save to temp
        temp_path = 'temp_uploads/temp_check_photo.jpg'
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        img.save(temp_path)
        # Call your quality check logic
        qualified = check_photo_quality(temp_path)
        return jsonify({'qualified': qualified})
    except Exception as e:
        return jsonify({'qualified': False, 'error': str(e)})

@app.route('/performance_stats', methods=['GET'])
def get_performance_stats():
    """Get system performance statistics"""
    try:
        success_rate = (performance_stats['successful_recognitions'] / performance_stats['total_recognitions'] * 100) if performance_stats['total_recognitions'] > 0 else 0
        failure_rate = (performance_stats['failed_recognitions'] / performance_stats['total_recognitions'] * 100) if performance_stats['total_recognitions'] > 0 else 0
        
        return jsonify({
            'success': True,
            'stats': {
                'total_recognitions': performance_stats['total_recognitions'],
                'successful_recognitions': performance_stats['successful_recognitions'],
                'failed_recognitions': performance_stats['failed_recognitions'],
                'success_rate': f"{success_rate:.1f}%",
                'failure_rate': f"{failure_rate:.1f}%",
                'average_confidence': f"{performance_stats['average_confidence']:.3f}",
                'registered_users': len(recognition_system.known_embeddings) if recognition_system else 0
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

# Cleanup function for graceful shutdown
def cleanup():
    """Clean up resources on shutdown"""
    try:
        cleanup_camera()
        print("Application shutdown complete")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup)

# Handle Windows socket errors gracefully
import signal
import sys

def signal_handler(sig, frame):
    print("\nReceived interrupt signal, shutting down gracefully...")
    cleanup()
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    try:
        print("Starting Face Recognition Application...")
        print("Checking for camera conflicts...")
        
        # Force release any existing camera connections
        force_release_all_cameras()
        time.sleep(3)  # Wait longer for cameras to be released
        
        print("Initializing camera...")
        camera_result = init_camera()
        if camera_result is None:
            print("⚠️  Warning: Camera initialization failed. Some features may not work.")
            print("💡 Try refreshing the page or restarting the application.")
        else:
            print("✅ Camera initialized successfully")
        print("Application ready!")
        
        # Use threaded mode to avoid socket issues on Windows
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down application...")
        cleanup()
    except Exception as e:
        print(f"Application error: {e}")
        cleanup()
    finally:
        cleanup()