# Face Recognition Web App - Complete Pipeline Documentation

## 📋 Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Application Startup](#application-startup)
3. [User Registration Pipeline](#user-registration-pipeline)
4. [Face Recognition Pipeline](#face-recognition-pipeline)
5. [Web Interface Routes](#web-interface-routes)
6. [Data Flow Diagrams](#data-flow-diagrams)
7. [Key Components](#key-components)

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask Web Application (app.py)            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  Registration    │         │   Recognition    │        │
│  │   System         │         │    System        │        │
│  └──────────────────┘         └──────────────────┘        │
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌──────────────────────────────────────────────┐         │
│  │     StandaloneRegistration / StandaloneRecognition │    │
│  │  - InsightFace RetinaFace (Face Detection)    │         │
│  │  - VGGFace2 InceptionResnetV1 (Embeddings)     │         │
│  │  - masktheface (Mask Augmentation)             │         │
│  └──────────────────────────────────────────────┘         │
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌──────────────────────────────────────────────┐         │
│  │         Embeddings Directory Structure        │         │
│  │  embeddings/                                   │         │
│  │    ├── {user_id}_{user_name}/                  │         │
│  │    │   ├── centroid.npy                       │         │
│  │    │   ├── metadata.json                      │         │
│  │    │   ├── originals/                         │         │
│  │    │   │   ├── original_1.npy                 │         │
│  │    │   │   ├── original_2.npy                 │         │
│  │    │   │   └── original_3.npy                 │         │
│  │    │   └── masked/                            │         │
│  │    │       ├── img_1/                         │         │
│  │    │       │   ├── surgical_blue.npy          │         │
│  │    │       │   ├── surgical_green.npy         │         │
│  │    │       │   ├── cloth.npy                  │         │
│  │    │       │   ├── n95.npy                    │         │
│  │    │       │   ├── kn95.npy                   │         │
│  │    │       │   └── gas.npy                    │         │
│  │    │       ├── img_2/ (same structure)       │         │
│  │    │       └── img_3/ (same structure)       │         │
│  └──────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Application Startup

### Step 1: Import and Initialize
```python
1. Load Flask framework and dependencies
2. Add project root to Python path
3. Try to import src.core modules (optional)
   - If available: Use FaceRegistration & FaceRecognition
   - If not: Use StandaloneRegistration & StandaloneRecognition
```

### Step 2: Initialize AI Models
```python
StandaloneRegistration:
  ├── InsightFace RetinaFace (Face Detection)
  │   └── Model: "buffalo_l"
  │   └── Input: 640x640 images
  │
  ├── VGGFace2 InceptionResnetV1 (Face Embeddings)
  │   └── Pretrained: 'vggface2'
  │   └── Output: 512-dimensional vectors
  │
  └── dlib (Mask Augmentation)
      └── Model: shape_predictor_68_face_landmarks.dat
      └── Used by masktheface library

StandaloneRecognition:
  ├── InsightFace RetinaFace (Face Detection)
  └── VGGFace2 InceptionResnetV1 (Face Embeddings)
```

### Step 3: Load Existing Embeddings
```python
1. Scan embeddings/ directory
2. For each user directory:
   - Load centroid.npy (average embedding)
   - Load all original embeddings from originals/
   - Load all masked embeddings from masked/img_X/
   - Load metadata.json (user info)
3. Filter out test entries (ID="img", numeric names)
4. Store in recognition_system.known_embeddings
```

### Step 4: Start Flask Server
```python
- Host: 0.0.0.0 (all interfaces)
- Port: 5000 (default)
- Debug: True (development mode)
```

---

## 👤 User Registration Pipeline

### Frontend Flow (registration.html)

```
User Input
    │
    ├── Name (text field)
    ├── Worker ID (text field)
    ├── Mobile Number (text field)
    └── 3 Images (file inputs: image1, image2, image3)
    │
    ▼
Form Validation (JavaScript)
    │
    ├── Check all fields filled
    ├── Validate image files
    └── Check image quality (optional)
    │
    ▼
POST /register (AJAX)
```

### Backend Flow (/register endpoint)

```
1. Receive Form Data
   ├── Extract: name, worker_id, mobile
   └── Extract: image1, image2, image3 files

2. Validation
   ├── Check required fields
   ├── Validate worker_id format (alphanumeric)
   ├── Validate mobile number format
   ├── Check all 3 images provided
   ├── Validate file sizes (< 10MB each)
   └── Validate file types (images)

3. Save Images Temporarily
   └── Save to temp_uploads/{worker_id}_{name}/

4. Call register_user_backend()
   │
   ▼
```

### Registration Processing (register_user_backend)

```
For each of 3 images:
│
├── Step 1: Quality Check
│   ├── Read image (cv2.imread)
│   ├── Calculate blur score (Laplacian variance)
│   └── Reject if blur < 40.0
│
├── Step 2: Face Detection
│   ├── Use InsightFace RetinaFace detector
│   ├── Get face bounding box
│   └── Extract 5-point facial landmarks
│
├── Step 3: Face Alignment
│   ├── Use ArcFace 5-point alignment
│   ├── Align to 112x112 pixels
│   └── Normalize lighting
│
├── Step 4: Extract Original Embedding
│   ├── Preprocess aligned face
│   │   ├── Convert to float32
│   │   ├── Normalize to [0, 1]
│   │   └── Apply normalization: (x - 0.5) / 0.5
│   ├── Pass through VGGFace2 model
│   ├── L2 normalize embedding
│   └── Save to originals/original_{idx}.npy
│
└── Step 5: Mask Augmentation (6 masks per image)
    │
    For each mask type (surgical_blue, surgical_green, cloth, n95, kn95, gas):
    │
    ├── Apply Mask
    │   ├── Detect face with dlib
    │   ├── Extract 68 facial landmarks
    │   ├── Get 6 key points for mask placement
    │   ├── Apply mask using masktheface
    │   └── Return masked image
    │
    ├── Re-detect Face (on masked image)
    │   └── Use InsightFace RetinaFace
    │
    ├── Re-align Face
    │   └── Use ArcFace 5-point alignment
    │
    ├── Extract Masked Embedding
    │   └── Same process as original embedding
    │
    └── Save to masked/img_{idx}/{mask_type}.npy

After all images processed:
│
├── Calculate Centroid
│   ├── Average all embeddings (3 originals + 18 masked = 21 total)
│   └── L2 normalize
│   └── Save to centroid.npy
│
├── Create Metadata
│   ├── user_id, user_name, mobile_number
│   ├── registration_date
│   ├── total_embeddings count
│   ├── quality_scores
│   └── recommended_threshold
│   └── Save to metadata.json
│
└── Return Success
    └── Reload recognition system embeddings
```

### Registration Output Structure

```
embeddings/{user_id}_{user_name}/
├── centroid.npy                    # Average of all 21 embeddings
├── metadata.json                   # User information
├── originals/
│   ├── original_1.npy             # Embedding from image 1
│   ├── original_2.npy             # Embedding from image 2
│   └── original_3.npy             # Embedding from image 3
└── masked/
    ├── img_1/
    │   ├── surgical_blue.npy      # Masked embedding 1
    │   ├── surgical_green.npy     # Masked embedding 2
    │   ├── cloth.npy              # Masked embedding 3
    │   ├── n95.npy                # Masked embedding 4
    │   ├── kn95.npy               # Masked embedding 5
    │   └── gas.npy                # Masked embedding 6
    ├── img_2/ (same 6 masks)
    └── img_3/ (same 6 masks)

Total: 3 originals + (6 masks × 3 images) = 21 embeddings per user
```

---

## 🔍 Face Recognition Pipeline

### Frontend Flow (realtime.html or image upload)

```
Option 1: Real-time Camera
    │
    ├── Access webcam (getUserMedia)
    ├── Capture frames (30 FPS)
    └── Send frames to /recognize_stream

Option 2: Image Upload
    │
    ├── User selects image file
    └── POST /recognize with image file
```

### Backend Flow (/recognize endpoint)

```
1. Receive Image
   ├── Read image file from request
   ├── Validate file size (< 10MB)
   ├── Decode image (cv2.imdecode)
   └── Validate dimensions (min 50x50)

2. Call recognize_user_backend(frame)
   │
   ▼
```

### Recognition Processing (recognize_user_backend)

```
Step 1: Face Detection
    ├── Use InsightFace RetinaFace
    ├── Detect faces in image
    └── Get best face (highest confidence)

Step 2: Extract Landmarks
    ├── Get 5-point facial landmarks
    └── Validate landmarks exist

Step 3: Face Alignment
    ├── Use ArcFace 5-point alignment
    ├── Align to 112x112 pixels
    └── Normalize lighting

Step 4: Extract Query Embedding
    ├── Preprocess aligned face
    ├── Pass through VGGFace2 model
    └── L2 normalize embedding

Step 5: Compare with Known Embeddings
    │
    For each registered user:
    │
    ├── Get all embeddings for user
    │   ├── Original embeddings (from originals/)
    │   └── Masked embeddings (from masked/img_X/)
    │
    ├── Calculate Similarities
    │   ├── Cosine similarity: dot(query_emb, known_emb)
    │   └── For each embedding of this user
    │
    ├── Find Maximum Similarity
    │   └── Take max similarity across all user's embeddings
    │
    └── Track Best Match
        ├── Store user with highest similarity
        └── Store similarity score

Step 6: Decision Making
    │
    ├── Threshold Check
    │   ├── Default threshold: 0.65
    │   ├── If best_score >= threshold:
    │   │   └── Return: user_name, "Present", similarity
    │   └── Else:
    │       └── Return: None, "Not in Database", similarity
    │
    └── Return Result
        ├── Display name
        ├── Status (Present/Not in Database)
        ├── Time stamp
        ├── Confidence score
        └── Bounding box (for visualization)
```

### Recognition Response

```json
{
  "success": true,
  "message": "Recognized with similarity 0.85",
  "name": "John Doe",
  "status": "Present",
  "time": "14:30:25",
  "bbox": [x, y, width, height]
}
```

---

## 🌐 Web Interface Routes

### Main Routes

| Route | Method | Purpose | Template |
|-------|--------|---------|----------|
| `/` | GET | Home/Dashboard | `index.html` |
| `/registration` | GET | Registration Form | `registration.html` |
| `/realtime` | GET | Real-time Recognition | `realtime.html` |
| `/user_access` | GET | User Management | `user_access.html` |
| `/expert_recognition` | GET | Expert Recognition | `expert_recognition.html` |

### API Endpoints

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/register` | POST | Register new user | Form data + 3 images | Success/Error |
| `/recognize` | POST | Recognize face | Image file | Name, Status, Time |
| `/recognize_stream` | GET | Real-time stream | Camera frames | Video stream |
| `/detect_face` | POST | Face detection only | Image file | Bounding box |
| `/check_photo` | POST | Photo quality check | Image file | Quality score |
| `/registered_users` | GET | Get user list | None | User list JSON |
| `/user_stats` | GET | Get statistics | None | Counts JSON |
| `/log_access` | POST | Log access attempt | JSON data | Success/Error |
| `/delete_user` | POST | Delete user | User ID | Success/Error |

---

## 📊 Data Flow Diagrams

### Registration Flow

```
User Browser
    │
    │ 1. Fill Form + Upload 3 Images
    ▼
Flask Server (/register)
    │
    │ 2. Validate & Save Images
    ▼
register_user_backend()
    │
    │ 3. Process Each Image
    ▼
StandaloneRegistration
    │
    ├── 4a. Detect Face (InsightFace)
    ├── 4b. Align Face (ArcFace)
    ├── 4c. Extract Embedding (VGGFace2)
    ├── 4d. Apply 6 Masks (masktheface)
    └── 4e. Extract Masked Embeddings
    │
    │ 5. Calculate Centroid
    ▼
Save to embeddings/{user_id}_{user_name}/
    │
    │ 6. Return Success
    ▼
Browser (Display Success Message)
```

### Recognition Flow

```
User Browser
    │
    │ 1. Upload Image or Camera Frame
    ▼
Flask Server (/recognize)
    │
    │ 2. Decode Image
    ▼
recognize_user_backend()
    │
    │ 3. Detect & Align Face
    ▼
StandaloneRecognition
    │
    ├── 4a. Detect Face (InsightFace)
    ├── 4b. Align Face (ArcFace)
    └── 4c. Extract Query Embedding (VGGFace2)
    │
    │ 5. Compare with Known Embeddings
    ▼
For each user:
    ├── Load all embeddings (originals + masked)
    ├── Calculate cosine similarities
    └── Find maximum similarity
    │
    │ 6. Decision (threshold check)
    ▼
Return Result (Name, Status, Confidence)
    │
    │ 7. Display Result
    ▼
Browser (Show Recognition Result)
```

---

## 🔧 Key Components

### 1. StandaloneRegistration Class

**Purpose**: Register new users with mask augmentation

**Key Methods**:
- `__init__()`: Initialize AI models
- `_align_face()`: Align face using 5-point landmarks
- `_extract_embedding()`: Extract face embedding
- `_apply_mask()`: Apply mask using masktheface
- `register_user()`: Main registration method

**Models Used**:
- InsightFace RetinaFace (detection)
- VGGFace2 InceptionResnetV1 (embeddings)
- dlib + masktheface (mask augmentation)

### 2. StandaloneRecognition Class

**Purpose**: Recognize registered users

**Key Methods**:
- `__init__()`: Initialize models and load embeddings
- `_load_embeddings()`: Load all user embeddings
- `_align_face()`: Align face using 5-point landmarks
- `_extract_embedding()`: Extract face embedding
- `recognize_image()`: Main recognition method

**Models Used**:
- InsightFace RetinaFace (detection)
- VGGFace2 InceptionResnetV1 (embeddings)

### 3. Embedding Structure

**Why 21 embeddings per user?**
- 3 original images → 3 original embeddings
- Each image gets 6 mask types → 6 masked embeddings per image
- Total: 3 + (6 × 3) = 21 embeddings

**Benefits**:
- Better recognition accuracy
- Handles mask-wearing scenarios
- More robust to variations

### 4. Similarity Calculation

**Method**: Cosine Similarity
```
similarity = dot(query_embedding, known_embedding)
           = |query| × |known| × cos(θ)

Since embeddings are L2-normalized:
similarity = cos(θ)  (ranges from -1 to 1)

For face recognition:
- similarity > 0.65: Likely match
- similarity > 0.70: Good match
- similarity > 0.80: Very confident match
```

**Matching Strategy**:
- Compare query embedding against ALL embeddings of each user
- Take maximum similarity across all embeddings
- This handles variations (masks, lighting, angles)

---

## 📝 Summary

### Registration Process
1. User uploads 3 images
2. System detects and aligns faces
3. Extracts 3 original embeddings
4. Applies 6 mask types to each image
5. Extracts 18 masked embeddings
6. Calculates centroid (average of all 21)
7. Saves to embeddings directory

### Recognition Process
1. User provides image (upload or camera)
2. System detects and aligns face
3. Extracts query embedding
4. Compares with all known embeddings
5. Finds best match (maximum similarity)
6. Checks against threshold (0.65)
7. Returns recognition result

### Key Features
- ✅ Mask augmentation for robust recognition
- ✅ Multiple embeddings per user (21 total)
- ✅ Real-time camera recognition
- ✅ Image upload recognition
- ✅ User management interface
- ✅ Access logging
- ✅ Statistics dashboard

---

## 🔄 Complete Request-Response Cycle

### Registration Example

```
1. User visits /registration
2. Fills form: Name="John", ID="123", Mobile="1234567890"
3. Uploads 3 images
4. JavaScript validates and sends POST /register
5. Server processes images (21 embeddings created)
6. Saves to embeddings/123_John/
7. Returns JSON: {"success": true, "message": "..."}
8. Browser shows success message
```

### Recognition Example

```
1. User visits /realtime or uploads image
2. Image sent to POST /recognize
3. Server detects face, extracts embedding
4. Compares with all known embeddings
5. Finds best match: "John" with similarity 0.85
6. Returns JSON: {"name": "John", "status": "Present", ...}
7. Browser displays recognition result
8. Access logged to user_access_log.csv
```

---

This pipeline ensures robust face recognition with mask augmentation, handling real-world scenarios where users may wear masks.

