#!/usr/bin/env python3
"""
Analyze PCA Visualization Images

Analyzes the PCA and t-SNE visualization images generated for embeddings.
Also provides detailed statistics about embedding structure and quality.
"""

import os
import numpy as np
import json

# Try to import visualization libraries, but make them optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available - will show basic file info only")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

def analyze_pca_images():
    """Analyze PCA visualization images in the directory"""
    
    print("=" * 70)
    print("PCA & t-SNE VISUALIZATION IMAGE ANALYSIS")
    print("=" * 70)
    
    # Check for visualization files
    pca_2d = "embeddings_pca_2d.png"
    pca_3d = "embeddings_pca_3d.png"
    tsne = "embeddings_tsne.png"
    
    files_found = []
    
    for filename in [pca_2d, pca_3d, tsne]:
        if os.path.exists(filename):
            files_found.append(filename)
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"\n✓ Found: {filename}")
            print(f"  Size: {file_size:.2f} KB")
            
            # Get image dimensions
            try:
                if HAS_PIL:
                    img = Image.open(filename)
                    width, height = img.size
                    print(f"  Dimensions: {width} x {height} pixels")
                    print(f"  Format: {img.format}")
                    print(f"  Mode: {img.mode}")
                elif HAS_MATPLOTLIB:
                    img = mpimg.imread(filename)
                    height, width = img.shape[:2]
                    print(f"  Dimensions: {width} x {height} pixels")
                    print(f"  Array shape: {img.shape}")
                else:
                    print(f"  (Install PIL or matplotlib for detailed image info)")
            except Exception as e:
                print(f"  ⚠️  Error reading image: {e}")
        else:
            print(f"\n✗ Not found: {filename}")
    
    print("\n" + "=" * 70)
    print("IMAGE DETAILS")
    print("=" * 70)
    
    # Analyze each image
    for filename in files_found:
        print(f"\n📊 Analyzing: {filename}")
        print("-" * 70)
        
        if HAS_MATPLOTLIB:
            try:
                img = mpimg.imread(filename)
                print(f"  Array shape: {img.shape}")
                print(f"  Data type: {img.dtype}")
                print(f"  Value range: [{img.min():.3f}, {img.max():.3f}]")
                
                # Check if it's a valid visualization
                if len(img.shape) == 3:  # RGB or RGBA
                    print(f"  Channels: {img.shape[2]}")
                    if img.shape[2] == 4:
                        print(f"  Has alpha channel (transparency)")
                
                # Try to display some info about what's in the image
                print(f"  ✓ Image loaded successfully")
                
            except Exception as e:
                print(f"  ✗ Error analyzing image: {e}")
        else:
            print(f"  (Install matplotlib for detailed analysis)")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION TYPES")
    print("=" * 70)
    
    if pca_2d in files_found:
        print("\n📉 PCA 2D (Principal Component Analysis - 2D)")
        print("  • Shows embeddings projected onto first 2 principal components")
        print("  • Useful for seeing overall clustering and separation")
        print("  • Similar embeddings cluster together")
        print("  • Different types (original/masked) should show patterns")
    
    if pca_3d in files_found:
        print("\n📊 PCA 3D (Principal Component Analysis - 3D)")
        print("  • Shows embeddings projected onto first 3 principal components")
        print("  • More detailed view than 2D")
        print("  • Better visualization of multi-dimensional relationships")
        print("  • Can rotate/interact with if opened in appropriate viewer")
    
    if tsne in files_found:
        print("\n🔍 t-SNE (t-Distributed Stochastic Neighbor Embedding)")
        print("  • Non-linear dimensionality reduction")
        print("  • Preserves local neighborhood structure")
        print("  • Good for finding clusters in high-dimensional data")
        print("  • Similar embeddings will be close together")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    
    print("""
What to look for in the visualizations:

1. CLUSTERING:
   ✓ Original embeddings should cluster together
   ✓ Masked embeddings should be near their corresponding originals
   ✓ Different mask types may form sub-clusters

2. SEPARATION:
   ✓ Different users should be well-separated
   ✓ Same user's embeddings should be close together

3. DISTRIBUTION:
   ✓ Well-distributed points = good embedding diversity
   ✓ Tight clusters = consistent embeddings
   ✓ Overlapping clusters = may indicate similar features

4. QUALITY INDICATORS:
   ✓ Tight clustering of same user = Good
   ✓ Clear separation between users = Good
   ✓ All embeddings in same area = May indicate low diversity
   ✓ Scattered embeddings = May indicate inconsistent quality
    """)
    
    # Check if embedding data exists
    print("=" * 70)
    print("RELATED DATA FILES")
    print("=" * 70)
    
    if os.path.exists("embedding_summary.csv"):
        print("\n✓ Found: embedding_summary.csv")
        print("  Contains summary statistics of embeddings")
    
    if os.path.exists("embeddings"):
        embed_dirs = [d for d in os.listdir("embeddings") if os.path.isdir(os.path.join("embeddings", d))]
        print(f"\n✓ Found embeddings directory with {len(embed_dirs)} user(s)")
        
        total_originals = 0
        total_masked = 0
        
        for user_dir in embed_dirs:
            user_path = os.path.join("embeddings", user_dir)
            npy_files = [f for f in os.listdir(user_path) if f.endswith('.npy')]
            subdirs = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
            
            # Count originals
            originals_dir = os.path.join(user_path, 'originals')
            originals_count = 0
            if os.path.exists(originals_dir):
                originals_count = len([f for f in os.listdir(originals_dir) if f.endswith('.npy')])
                total_originals += originals_count
            
            # Count masked embeddings (new structure)
            masked_dir = os.path.join(user_path, 'masked')
            masked_count = 0
            if os.path.exists(masked_dir):
                for img_subdir in os.listdir(masked_dir):
                    img_subdir_path = os.path.join(masked_dir, img_subdir)
                    if os.path.isdir(img_subdir_path):
                        masked_count += len([f for f in os.listdir(img_subdir_path) if f.endswith('.npy')])
                total_masked += masked_count
            
            # Load metadata if available
            meta_path = os.path.join(user_path, 'metadata.json')
            metadata_info = ""
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    threshold = meta.get('recommended_threshold', 'N/A')
                    total_embs = meta.get('total_embeddings', 'N/A')
                    metadata_info = f" | Threshold: {threshold} | Total embeddings: {total_embs}"
                except:
                    pass
            
            print(f"  - {user_dir}:")
            print(f"    • {len(npy_files)} .npy files in root")
            print(f"    • {len(subdirs)} subdirectories")
            if originals_count > 0:
                print(f"    • {originals_count} original embeddings")
            if masked_count > 0:
                print(f"    • {masked_count} masked embeddings")
            if metadata_info:
                print(f"    {metadata_info}")
        
        print(f"\n📊 Total Statistics:")
        print(f"  • Total original embeddings: {total_originals}")
        print(f"  • Total masked embeddings: {total_masked}")
        print(f"  • Total embeddings: {total_originals + total_masked}")
    
    print("\n" + "=" * 70)
    print("EMBEDDING QUALITY CHECK")
    print("=" * 70)
    
    # Check embedding quality indicators
    if os.path.exists("embeddings"):
        embed_dirs = [d for d in os.listdir("embeddings") if os.path.isdir(os.path.join("embeddings", d))]
        
        if len(embed_dirs) >= 2:
            print("\n✓ Multiple users detected - can perform cross-user analysis")
            print("  Run visualize_embeddings.py to see similarity analysis")
        elif len(embed_dirs) == 1:
            print("\n⚠️  Only one user registered")
            print("  Register more users to see cross-user separation analysis")
        else:
            print("\n⚠️  No users registered")
            print("  Run register_mask_aug.py to register users")
        
        # Check for mask augmentation
        has_masked = False
        for user_dir in embed_dirs:
            user_path = os.path.join("embeddings", user_dir)
            masked_dir = os.path.join(user_path, 'masked')
            if os.path.exists(masked_dir) and os.listdir(masked_dir):
                has_masked = True
                break
        
        if has_masked:
            print("\n✓ Mask augmentation detected")
            print("  Masked embeddings are included in visualizations")
        else:
            print("\n⚠️  No masked embeddings found")
            print("  Mask augmentation helps with recognition accuracy")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✓ Found {len(files_found)} visualization file(s)")
    print(f"  These visualizations show your embedding space structure")
    print(f"  Use them to verify embedding quality and clustering")
    
    if len(files_found) < 3:
        print(f"\n⚠️  Missing {3 - len(files_found)} visualization file(s)")
        print(f"  Run: python visualize_embeddings.py")
        print(f"  This will generate all missing visualizations")
    
    print("\n" + "=" * 70)
    print("QUICK COMMANDS")
    print("=" * 70)
    print("\nTo regenerate visualizations:")
    print("  python visualize_embeddings.py")
    print("\nTo check embedding similarities:")
    print("  python check_user_similarities.py <user_id>")
    print("\nTo update recognition threshold:")
    print("  python update_threshold.py <user_id> <threshold>")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_pca_images()

