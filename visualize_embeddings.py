#!/usr/bin/env python3
"""
PCA Visualization for Face Embeddings
- Loads all user embeddings
- Applies PCA for 2D/3D visualization
- Shows embedding distribution and separation
- Interactive plot with user labels
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from collections import defaultdict
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_embeddings(embeddings_dir="embeddings"):
    """Load all embeddings from the embeddings directory"""
    print(f"📂 Loading embeddings from {embeddings_dir}...")
    
    all_embeddings = []
    user_labels = []
    user_info = {}
    embedding_details = []
    
    if not os.path.exists(embeddings_dir):
        print(f"❌ Embeddings directory not found: {embeddings_dir}")
        return None, None, None, None
    
    for user_dir in os.listdir(embeddings_dir):
        user_path = os.path.join(embeddings_dir, user_dir)
        if not os.path.isdir(user_path):
            continue
        
        # Extract user info from directory name
        parts = user_dir.split('_', 1)
        if len(parts) >= 2:
            user_id, user_name = parts[0], parts[1]
        else:
            user_id, user_name = user_dir, user_dir
        
        print(f"\n👤 User: {user_name} (ID: {user_id})")
        
        # Load centroid if available
        centroid_path = os.path.join(user_path, 'centroid.npy')
        if os.path.exists(centroid_path):
            centroid = np.load(centroid_path)
            all_embeddings.append(centroid)
            user_labels.append(user_name)
            embedding_details.append({
                'user_id': user_id,
                'user_name': user_name,
                'type': 'centroid',
                'file': 'centroid.npy'
            })
            print(f"  ✓ Centroid loaded (dim: {len(centroid)})")
        
        # Load original embeddings
        originals_dir = os.path.join(user_path, 'originals')
        if os.path.exists(originals_dir):
            original_files = [f for f in os.listdir(originals_dir) if f.endswith('.npy')]
            for i, orig_file in enumerate(original_files):
                orig_path = os.path.join(originals_dir, orig_file)
                embedding = np.load(orig_path)
                all_embeddings.append(embedding)
                user_labels.append(f"{user_name}_orig_{i+1}")
                embedding_details.append({
                    'user_id': user_id,
                    'user_name': user_name,
                    'type': 'original',
                    'file': orig_file,
                    'index': i+1
                })
            print(f"  ✓ {len(original_files)} original embeddings loaded")
        
        # Load masked embeddings (structure: masked/img_{idx}/{mask_type}.npy)
        masked_dir = os.path.join(user_path, 'masked')
        if os.path.exists(masked_dir):
            masked_count = 0
            for img_subdir in sorted(os.listdir(masked_dir)):
                img_subdir_path = os.path.join(masked_dir, img_subdir)
                if not os.path.isdir(img_subdir_path):
                    continue
                
                # Load all mask types from this image subdirectory
                for mask_file in sorted(os.listdir(img_subdir_path)):
                    if mask_file.endswith('.npy'):
                        mask_path = os.path.join(img_subdir_path, mask_file)
                        embedding = np.load(mask_path)
                        all_embeddings.append(embedding)
                        mask_type = mask_file.replace('.npy', '')
                        user_labels.append(f"{user_name}_{mask_type}")
                        embedding_details.append({
                            'user_id': user_id,
                            'user_name': user_name,
                            'type': 'masked',
                            'file': mask_file,
                            'mask_type': mask_type,
                            'img_dir': img_subdir
                        })
                        masked_count += 1
            if masked_count > 0:
                print(f"  ✓ {masked_count} masked embeddings loaded")
        
        # Load any other .npy files in user directory
        other_files = [f for f in os.listdir(user_path) if f.endswith('.npy') and f != 'centroid.npy']
        for i, other_file in enumerate(other_files):
            other_path = os.path.join(user_path, other_file)
            embedding = np.load(other_path)
            all_embeddings.append(embedding)
            user_labels.append(f"{user_name}_other_{i+1}")
            embedding_details.append({
                'user_id': user_id,
                'user_name': user_name,
                'type': 'other',
                'file': other_file,
                'index': i+1
            })
        if other_files:
            print(f"  ✓ {len(other_files)} other embeddings loaded")
        
        user_info[user_name] = {
            'id': user_id,
            'total_embeddings': len([d for d in embedding_details if d['user_name'] == user_name])
        }
    
    if len(all_embeddings) == 0:
        print("❌ No embeddings found")
        return None, None, None, None
    
    all_embeddings = np.array(all_embeddings)
    print(f"\n✅ Total embeddings loaded: {len(all_embeddings)}")
    print(f"✅ Embedding dimension: {all_embeddings.shape[1]}")
    print(f"✅ Users: {len(user_info)}")
    
    return all_embeddings, user_labels, user_info, embedding_details

def analyze_embedding_quality(embeddings, labels, user_info):
    """Analyze embedding quality and separation"""
    print(f"\n📊 EMBEDDING QUALITY ANALYSIS")
    print(f"{'='*50}")
    
    # Basic statistics
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Mean norm: {np.mean([np.linalg.norm(e) for e in embeddings]):.6f}")
    print(f"Std norm: {np.std([np.linalg.norm(e) for e in embeddings]):.6f}")
    
    # Check for NaN/Inf
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    if has_nan:
        print("❌ NaN values found in embeddings!")
    if has_inf:
        print("❌ Inf values found in embeddings!")
    if not has_nan and not has_inf:
        print("✓ No NaN or Inf values")
    
    # Value range
    print(f"Value range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
    print(f"Mean: {embeddings.mean():.6f}")
    print(f"Std: {embeddings.std():.6f}")
    
    # Calculate intra-class and inter-class similarities
    unique_users = list(user_info.keys())
    if len(unique_users) < 2:
        print("\n⚠️  Need at least 2 users for similarity analysis")
        print("   (Only one user registered, no cross-user comparison)")
        return
    
    intra_similarities = []
    inter_similarities = []
    
    for user in unique_users:
        user_embeddings = [embeddings[i] for i, label in enumerate(labels) if label.startswith(user)]
        
        if len(user_embeddings) < 2:
            continue
        
        # Same person similarities (cosine similarity, not distance)
        for i in range(len(user_embeddings)):
            for j in range(i + 1, len(user_embeddings)):
                sim = np.dot(user_embeddings[i], user_embeddings[j]) / (
                    np.linalg.norm(user_embeddings[i]) * np.linalg.norm(user_embeddings[j]) + 1e-12
                )
                intra_similarities.append(float(sim))
        
        # Different person similarities
        for other_user in unique_users:
            if other_user == user:
                continue
            
            other_embeddings = [embeddings[i] for i, label in enumerate(labels) if label.startswith(other_user)]
            if len(other_embeddings) == 0:
                continue
            
            # Compare all embeddings between users for better statistics
            for emb1 in user_embeddings:
                for emb2 in other_embeddings:
                    sim = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-12
                    )
                    inter_similarities.append(float(sim))
    
    if len(intra_similarities) > 0 and len(inter_similarities) > 0:
        intra_similarities = np.array(intra_similarities)
        inter_similarities = np.array(inter_similarities)
        
        print(f"\n🔍 Similarity Analysis:")
        print(f"  Intra-class (same person):")
        print(f"    Count: {len(intra_similarities)}")
        print(f"    Mean: {np.mean(intra_similarities):.4f}")
        print(f"    Std: {np.std(intra_similarities):.4f}")
        print(f"    Min: {np.min(intra_similarities):.4f}")
        print(f"    Max: {np.max(intra_similarities):.4f}")
        
        print(f"\n  Inter-class (different persons):")
        print(f"    Count: {len(inter_similarities)}")
        print(f"    Mean: {np.mean(inter_similarities):.4f}")
        print(f"    Std: {np.std(inter_similarities):.4f}")
        print(f"    Min: {np.min(inter_similarities):.4f}")
        print(f"    Max: {np.max(inter_similarities):.4f}")
        
        # Separation quality (higher intra, lower inter = better)
        # For cosine similarity: higher = more similar
        # Good separation: intra-class similarities should be HIGH, inter-class should be LOW
        separation = np.mean(intra_similarities) - np.mean(inter_similarities)
        print(f"\n  Separation: {separation:.4f}")
        print(f"    (Higher is better: intra-class should be > inter-class)")
        
        if separation > 0.3:
            print("  ✅ Excellent separation")
        elif separation > 0.2:
            print("  ✅ Good separation")
        elif separation > 0.1:
            print("  ⚠️  Moderate separation")
        elif separation > 0:
            print("  ⚠️  Weak separation")
        else:
            print("  ❌ Poor separation (inter-class > intra-class)")

def visualize_pca_2d(embeddings, labels, user_info, save_plot=True):
    """Create 2D PCA visualization"""
    print(f"\n🎨 Creating 2D PCA visualization...")
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels,
        'user': [label.split('_')[0] for label in labels]
    })
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each user with different colors
    unique_users = df['user'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_users)))
    
    for i, user in enumerate(unique_users):
        user_data = df[df['user'] == user]
        
        # Plot points
        plt.scatter(user_data['x'], user_data['y'], 
                   c=[colors[i]], label=user, alpha=0.7, s=60)
        
        # Add user name annotation
        if len(user_data) > 0:
            center_x = user_data['x'].mean()
            center_y = user_data['y'].mean()
            plt.annotate(user, (center_x, center_y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Face Embeddings - 2D PCA Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig('embeddings_pca_2d.png', dpi=300, bbox_inches='tight')
        print("✓ 2D PCA plot saved as 'embeddings_pca_2d.png'")
    
    plt.show()

def visualize_pca_3d(embeddings, labels, user_info, save_plot=True):
    """Create 3D PCA visualization"""
    print(f"\n🎨 Creating 3D PCA visualization...")
    
    # Apply PCA
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Create plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each user with different colors
    unique_users = list(set([label.split('_')[0] for label in labels]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_users)))
    
    for i, user in enumerate(unique_users):
        user_indices = [j for j, label in enumerate(labels) if label.startswith(user)]
        user_embeddings_3d = embeddings_3d[user_indices]
        
        ax.scatter(user_embeddings_3d[:, 0], user_embeddings_3d[:, 1], user_embeddings_3d[:, 2],
                  c=[colors[i]], label=user, alpha=0.7, s=60)
        
        # Add user name annotation
        if len(user_embeddings_3d) > 0:
            center = user_embeddings_3d.mean(axis=0)
            ax.text(center[0], center[1], center[2], user, fontsize=10, fontweight='bold')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
    ax.set_title('Face Embeddings - 3D PCA Visualization')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_plot:
        plt.savefig('embeddings_pca_3d.png', dpi=300, bbox_inches='tight')
        print("✓ 3D PCA plot saved as 'embeddings_pca_3d.png'")
    
    plt.show()

def visualize_tsne(embeddings, labels, user_info, save_plot=True):
    """Create t-SNE visualization"""
    print(f"\n🎨 Creating t-SNE visualization...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': embeddings_tsne[:, 0],
        'y': embeddings_tsne[:, 1],
        'label': labels,
        'user': [label.split('_')[0] for label in labels]
    })
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each user with different colors
    unique_users = df['user'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_users)))
    
    for i, user in enumerate(unique_users):
        user_data = df[df['user'] == user]
        
        # Plot points
        plt.scatter(user_data['x'], user_data['y'], 
                   c=[colors[i]], label=user, alpha=0.7, s=60)
        
        # Add user name annotation
        if len(user_data) > 0:
            center_x = user_data['x'].mean()
            center_y = user_data['y'].mean()
            plt.annotate(user, (center_x, center_y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Face Embeddings - t-SNE Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig('embeddings_tsne.png', dpi=300, bbox_inches='tight')
        print("✓ t-SNE plot saved as 'embeddings_tsne.png'")
    
    plt.show()

def create_embedding_summary(embedding_details, user_info):
    """Create a summary table of embeddings"""
    print(f"\n📋 EMBEDDING SUMMARY")
    print(f"{'='*80}")
    
    # Create summary DataFrame
    df = pd.DataFrame(embedding_details)
    
    # Group by user and type
    summary = df.groupby(['user_name', 'type']).size().unstack(fill_value=0)
    
    print("Embeddings per user and type:")
    print(summary)
    
    # Save summary to CSV
    summary.to_csv('embedding_summary.csv')
    print(f"\n✓ Summary saved to 'embedding_summary.csv'")
    
    # Detailed breakdown
    print(f"\nDetailed breakdown:")
    for user in user_info.keys():
        user_data = df[df['user_name'] == user]
        print(f"\n👤 {user}:")
        
        type_counts = user_data['type'].value_counts()
        for emb_type, count in type_counts.items():
            print(f"  {emb_type}: {count} embeddings")

def main():
    parser = argparse.ArgumentParser(description='PCA Visualization for Face Embeddings')
    parser.add_argument('--embeddings-dir', default='embeddings', help='Embeddings directory')
    parser.add_argument('--method', choices=['pca2d', 'pca3d', 'tsne', 'all'], default='all',
                       help='Visualization method')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"FACE EMBEDDING VISUALIZATION")
    print(f"{'='*70}")
    
    # Load embeddings
    embeddings, labels, user_info, embedding_details = load_all_embeddings(args.embeddings_dir)
    
    if embeddings is None:
        print("❌ Failed to load embeddings")
        return
    
    # Analyze quality
    analyze_embedding_quality(embeddings, labels, user_info)
    
    # Create summary
    create_embedding_summary(embedding_details, user_info)
    
    # Create visualizations
    save_plot = not args.no_save
    
    if args.method in ['pca2d', 'all']:
        visualize_pca_2d(embeddings, labels, user_info, save_plot)
    
    if args.method in ['pca3d', 'all']:
        visualize_pca_3d(embeddings, labels, user_info, save_plot)
    
    if args.method in ['tsne', 'all']:
        visualize_tsne(embeddings, labels, user_info, save_plot)
    
    print(f"\n✅ Visualization complete!")

if __name__ == "__main__":
    main()
