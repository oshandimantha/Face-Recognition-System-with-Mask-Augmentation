# Face-Recognition-System-with-Mask-Augmentation
A comprehensive face recognition system that supports mask augmentation for improved recognition accuracy. This system uses deep learning models to register and recognize faces, with special support for masked face scenarios.

## 🎯 Features

- 🎭 **Mask Augmentation**: Automatically augments face images with various mask types for training
- 👤 **Face Registration**: Register new users with multiple images via web interface or CLI
- 🔍 **Face Recognition**: Real-time and batch face recognition capabilities
- 📊 **Web Interface**: Flask-based web application for easy interaction
- 📈 **Visualization**: PCA and t-SNE visualization of face embeddings
- 🔐 **Access Logging**: Track user access and recognition events
- 🎨 **Multiple Mask Types**: Support for 6+ different mask types (N95, surgical, cloth, etc.)
- 🚀 **GPU Support**: Optional CUDA support for faster processing

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Storage**: 500 MB free space
- **GPU**: Optional (CUDA-compatible GPU recommended for better performance)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/face07.git
cd face07
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install dlib (Windows - if needed)

If dlib installation fails on Windows, use the provided wheel file:

```bash
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
```

Or download from: https://github.com/sachadee/Dlib

### Step 5: Download Required Models

- **InsightFace models**: Will be auto-downloaded on first use (stored in `~/.insightface/models/`)
- **VGGFace2 model**: Will be auto-downloaded on first use (cached by facenet-pytorch)
- **dlib shape predictor**: Should be in `dlib_models/` directory

### Step 6: Verify Installation

```bash
# Test imports
python -c "import cv2, numpy, torch, insightface, facenet_pytorch, dlib, flask; print('✅ All imports successful!')"
```

## 💻 Usage

### Running the Web Application

```bash
python app.py
```

Then open your browser and navigate to:
- **Main Interface**: `http://localhost:5000`
- **Registration**: `http://localhost:5000/registration`
- **Recognition**: `http://localhost:5000/realtime`

### Command Line Usage

#### Register a new user:
```bash
python register_mask_aug.py
# Or with arguments:
python register_mask_aug.py --name "John Doe" --id "123"
```

#### Recognize faces from camera:
```bash
python recognize_mask_aug.py --cam 0
```

#### Recognize faces from image:
```bash
python recognize_mask_aug.py --image path/to/test.jpg
```

#### Visualize embeddings:
```bash
python visualize_embeddings.py
```

## 📁 Project Structure

```
face07/
├── app.py                      # Flask web application (main entry point)
├── register_mask_aug.py        # CLI registration script
├── recognize_mask_aug.py        # CLI recognition script
├── visualize_embeddings.py     # Embedding visualization tool
├── analyze_pca_images.py       # PCA analysis script
├── check_embedding_details.py  # Embedding inspection tool
├── update_threshold.py         # Threshold adjustment tool
├── recalculate_treshlod.py     # Threshold recalculation
│
├── config/                     # Configuration files
│   ├── __init__.py
│   └── settings.py
│
├── templates/                  # HTML templates
│   ├── index.html
│   ├── registration.html
│   ├── realtime.html
│   ├── user.html
│   ├── user_access.html
│   └── expert_recognition.html
│
├── static/                     # Static assets
│   ├── style.css
│   ├── main.js
│   └── *.png, *.jpg
│
├── embeddings/                 # Stored face embeddings
│   ├── {user_id}_{name}/
│   │   ├── centroid.npy
│   │   ├── metadata.json
│   │   ├── originals/
│   │   ├── masked/
│   │   └── visualizations/
│
├── masks/                      # Mask templates and configurations
│   ├── masks.cfg
│   ├── templates/
│   └── templates_with_points/
│
├── masktheface/                # Mask augmentation library
│   ├── mask_the_face.py
│   ├── utils/
│   └── masks/
│
├── dlib_models/                # dlib model files
│   └── shape_predictor_68_face_landmarks.dat
│
├── data/                       # Application data
│   └── db.json
│
├── temp_uploads/               # Temporary upload directory
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── INSTALLATION_GUIDE.md       # Detailed installation guide
├── TECHNICAL_REPORT.md         # Technical documentation
└── WEB_APP_PIPELINE.md         # Web app architecture
```

## 🛠️ Technologies Used

- **Flask**: Web framework for the application interface
- **OpenCV**: Image processing and computer vision
- **PyTorch**: Deep learning framework
- **InsightFace**: Face detection and recognition models
- **FaceNet-PyTorch**: Face embedding extraction (VGGFace2)
- **dlib**: Face detection and landmark detection for mask augmentation
- **scikit-learn**: Machine learning utilities (PCA, t-SNE)
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization

## 📚 Documentation

- **[Installation Guide](INSTALLATION_GUIDE.md)**: Detailed installation instructions
- **[Technical Report](TECHNICAL_REPORT.md)**: Technical documentation and architecture
- **[Web App Pipeline](WEB_APP_PIPELINE.md)**: Web application architecture and data flow

## 🔧 Configuration

### Threshold Settings

The recognition threshold can be adjusted in:
- `config/settings.py` - Default threshold values
- `update_threshold.py` - Script to update thresholds
- Web interface - Real-time threshold adjustment

### Mask Types

Available mask types (configured in `masks/masks.cfg`):
- N95
- Surgical
- Cloth
- Gas
- And more...

## 🐛 Troubleshooting

### dlib Installation Issues (Windows)
```bash
# Use the provided wheel file
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
```

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch if needed
# Visit: https://pytorch.org/get-started/locally/
```

### Model Download Issues
- InsightFace models are downloaded automatically on first use
- If download fails, check internet connection
- Models are cached in `~/.insightface/models/`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

[Add your license here - e.g., MIT License, Apache 2.0, etc.]

## 👤 Author

[Your Name]

## 🙏 Acknowledgments

- Mask augmentation based on [MaskTheFace](https://github.com/aqeelanwar/MaskTheFace)
- Face recognition powered by [InsightFace](https://github.com/deepinsight/insightface)
- Face embeddings using [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch)
- dlib for face landmark detection

## 📧 Contact

For questions or support, please open an issue on GitHub.
