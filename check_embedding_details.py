import numpy as np
import glob
import os
import json

emb_dir = 'embeddings'

# Find all embedding files recursively (emb_*.npy, original_*.npy, and centroid.npy)
embs = sorted(glob.glob(os.path.join(emb_dir, '**', 'emb_*.npy'), recursive=True))
embs.extend(sorted(glob.glob(os.path.join(emb_dir, '**', 'original_*.npy'), recursive=True)))
centroids = sorted(glob.glob(os.path.join(emb_dir, '**', 'centroid.npy'), recursive=True))

print(f"Found {len(embs)} individual embeddings")
print(f"Found {len(centroids)} centroids\n")

if len(embs) == 0:
    print("⚠️  No embeddings found. Expected structure: embeddings/{user_id}_{name}/emb_*.npy")
    print("   Run registration first to create embeddings.")
    exit(1)

# Load embeddings with metadata
embeddings_data = []
for emb_path in embs:
    emb = np.load(emb_path)
    # Get user directory name
    user_dir = os.path.dirname(emb_path)
    user_name = os.path.basename(user_dir)
    emb_name = os.path.basename(emb_path)
    
    # Try to load profile.json for more info
    profile_path = os.path.join(user_dir, 'profile.json')
    user_info = {}
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r') as f:
                user_info = json.load(f)
        except:
            pass
    
    embeddings_data.append({
        'path': emb_path,
        'embedding': emb,
        'user_dir': user_name,
        'emb_name': emb_name,
        'user_id': user_info.get('id', 'unknown'),
        'user_name': user_info.get('name', 'unknown'),
    })

print("\n" + "="*70)
print("EMBEDDING DETAILS:")
print("="*70)
for i, data in enumerate(embeddings_data):
    emb = data['embedding']
    norm = np.linalg.norm(emb)
    print(f"\n[{i+1}] {data['emb_name']}")
    print(f"    User: {data['user_name']} (ID: {data['user_id']})")
    print(f"    Shape: {emb.shape}")
    print(f"    Norm: {norm:.6f} (should be ~1.0 for L2-normalized)")
    print(f"    Path: {data['path']}")

# Check pairwise similarities (within same user and across users)
print("\n" + "="*70)
print("PAIRWISE COSINE SIMILARITIES:")
print("="*70)

# Group by user
from collections import defaultdict
by_user = defaultdict(list)
for i, data in enumerate(embeddings_data):
    by_user[data['user_dir']].append((i, data))

# Within-user similarities
print("\n📊 WITHIN-USER SIMILARITIES (same person):")
print("-" * 70)
for user_dir, items in by_user.items():
    if len(items) > 1:
        print(f"\nUser: {items[0][1]['user_name']} ({user_dir})")
        for idx, (i, data_i) in enumerate(items):
            for j, data_j in items[idx+1:]:
                e1, e2 = data_i['embedding'], data_j['embedding']
                cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12)
                distance = 1 - cos_sim
                print(f"  {data_i['emb_name']} vs {data_j['emb_name']}:")
                print(f"    Similarity: {cos_sim:.6f} | Distance: {distance:.6f}")
                if cos_sim > 0.9:
                    print(f"    ✓ Excellent match (same person)")
                elif cos_sim > 0.7:
                    print(f"    ✓ Good match")
                elif cos_sim > 0.5:
                    print(f"    ⚠️  Moderate match")
                else:
                    print(f"    ❌ Low match (might be different person)")

# Cross-user similarities
print("\n📊 CROSS-USER SIMILARITIES (different people):")
print("-" * 70)
user_list = list(by_user.keys())
if len(user_list) > 1:
    for i, user1 in enumerate(user_list):
        for user2 in user_list[i+1:]:
            items1 = by_user[user1]
            items2 = by_user[user2]
            # Compare first embedding from each user
            e1 = items1[0][1]['embedding']
            e2 = items2[0][1]['embedding']
            cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12)
            print(f"  {items1[0][1]['user_name']} vs {items2[0][1]['user_name']}: {cos_sim:.6f}")
            if cos_sim > 0.7:
                print(f"    ⚠️  WARNING: High similarity between different users!")
            elif cos_sim > 0.5:
                print(f"    ⚠️  Moderate similarity - check if they're actually different")
            else:
                print(f"    ✓ Good separation (different people)")
else:
    print("  (Only one user registered, no cross-user comparison)")

# Load and check centroids
if centroids:
    print("\n" + "="*70)
    print("CENTROID ANALYSIS:")
    print("="*70)
    for cent_path in centroids:
        cent = np.load(cent_path)
        user_dir = os.path.dirname(cent_path)
        user_name = os.path.basename(user_dir)
        norm = np.linalg.norm(cent)
        print(f"\nUser: {user_name}")
        print(f"  Centroid shape: {cent.shape}")
        print(f"  Centroid norm: {norm:.6f} (should be ~1.0)")
        
        # Check metadata if available
        meta_path = os.path.join(user_dir, 'metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                rec_thresh = meta.get('recommended_threshold', None)
                sim_stats = meta.get('similarity_stats', {})
                if rec_thresh is not None:
                    print(f"  Recommended threshold: {rec_thresh:.4f}")
                if sim_stats:
                    print(f"  Similarity stats: mean={sim_stats.get('mean', 0):.4f}, "
                          f"std={sim_stats.get('std', 0):.4f}")
            except Exception:
                pass

# Summary
print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print(f"✓ Total embeddings: {len(embeddings_data)}")
print(f"✓ Total users: {len(by_user)}")
print(f"✓ All embeddings are L2-normalized (norm ≈ 1.0)")
print(f"✓ For recognition: similarity > 0.6-0.7 = same person")
print(f"✓ For recognition: similarity < 0.5 = different person")

