# Installation Guide - Face Recognition System

## Quick Start

### 1. Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Storage**: 500 MB free space

### 2. Installation Steps

#### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

#### Step 3: Install dlib (Windows - if needed)

If dlib installation fails, use the provided wheel file:

```bash
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
```

Or download from: https://github.com/sachadee/Dlib

#### Step 4: Verify Installation

```bash
# Test imports
python -c "import cv2, numpy, torch, insightface, facenet_pytorch, dlib; print('All imports successful!')"
```

### 3. Model Downloads

Models are automatically downloaded on first use:

- **InsightFace RetinaFace**: Downloaded to `~/.insightface/models/buffalo_l/`
- **VGGFace2**: Downloaded by facenet-pytorch (cached automatically)
- **dlib Landmark Predictor**: Should be in `dlib_models/shape_predictor_68_face_landmarks.dat`

### 4. GPU Support (Optional)

For faster performance with NVIDIA GPU:

1. Install CUDA Toolkit 11.8 or higher
2. Install CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### 5. Verify System

```bash
# Check if GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test registration
python register_mask_aug.py --name test

# Test recognition
python recognize_mask_aug_fast.py --fast
```

## Troubleshooting

### Issue: dlib installation fails
**Solution**: Use the provided wheel file or install Visual C++ Build Tools

### Issue: CUDA not detected
**Solution**: Install CUDA toolkit and reinstall PyTorch with CUDA support

### Issue: Models not downloading
**Solution**: Check internet connection, models download on first use

### Issue: Import errors
**Solution**: Ensure virtual environment is activated and all packages are installed

## System Ready!

Once installation is complete, you can:
- Register users: `python register_mask_aug.py --name "User Name"`
- Recognize faces: `python recognize_mask_aug_fast.py`
- Visualize embeddings: `python visualize_embeddings.py`

