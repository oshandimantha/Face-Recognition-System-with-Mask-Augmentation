#!/usr/bin/env python3
"""
Recalculate thresholds for all users based on cross-user analysis
This ensures thresholds adapt as more users are added to the system
"""

import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def load_user_embeddings(user_dir):
    """Load all embeddings for a user"""
    embs = []
    
    # Load originals
    orig_dir = os.path.join(user_dir, 'originals')
    if os.path.exists(orig_dir):
        for f in sorted(os.listdir(orig_dir)):
            if f.endswith('.npy'):
                embs.append(np.load(os.path.join(orig_dir, f)))
    
    # Load masked
    masked_dir = os.path.join(user_dir, 'masked')
    if os.path.exists(masked_dir):
        for img_sub in sorted(os.listdir(masked_dir)):
            sub = os.path.join(masked_dir, img_sub)
            if os.path.isdir(sub):
                for f in sorted(os.listdir(sub)):
                    if f.endswith('.npy'):
                        embs.append(np.load(os.path.join(sub, f)))
    
    return embs

def calculate_optimal_threshold(user_embs, all_other_embs, min_threshold=0.5, max_threshold=0.9):
    """
    Calculate optimal threshold for a user considering:
    1. Intra-user similarity (same person)
    2. Inter-user similarity (different persons)
    """
    if not user_embs:
        return 0.7
    
    # Intra-user similarities (same person)
    intra_sims = []
    for i in range(len(user_embs)):
        for j in range(i + 1, len(user_embs)):
            intra_sims.append(cosine_similarity(user_embs[i], user_embs[j]))
    
    if not intra_sims:
        return 0.7
    
    intra_mean = np.mean(intra_sims)
    intra_std = np.std(intra_sims)
    intra_min = np.min(intra_sims)
    
    # Inter-user similarities (different persons)
    inter_sims = []
    if all_other_embs:
        for user_emb in user_embs:
            for other_emb in all_other_embs:
                inter_sims.append(cosine_similarity(user_emb, other_emb))
    
    if inter_sims:
        inter_max = np.max(inter_sims)
        inter_mean = np.mean(inter_sims)
        
        # Strategy: Set threshold between intra_min and inter_max
        # Use: intra_min - safety_margin, but ensure it's above inter_max
        safety_margin = 0.05
        optimal_threshold = max(inter_max + 0.1, intra_min - safety_margin)
        
        # Clamp to reasonable range
        optimal_threshold = max(min_threshold, min(max_threshold, optimal_threshold))
        
        print(f"    Intra-user: min={intra_min:.3f}, mean={intra_mean:.3f}, std={intra_std:.3f}")
        print(f"    Inter-user: max={inter_max:.3f}, mean={inter_mean:.3f}")
        print(f"    Optimal threshold: {optimal_threshold:.3f} (ensures separation from others)")
    else:
        # No other users - use original method
        optimal_threshold = max(min_threshold, min(max_threshold, intra_mean - 2.0 * intra_std))
        print(f"    Intra-user only: mean={intra_mean:.3f}, std={intra_std:.3f}")
        print(f"    Threshold: {optimal_threshold:.3f}")
    
    return optimal_threshold

def main():
    print("="*70)
    print("AUTOMATIC THRESHOLD RECALCULATION")
    print("="*70)
    print("\nThis script recalculates thresholds for all users based on:")
    print("  • Intra-user similarities (same person)")
    print("  • Inter-user similarities (different persons)")
    print("  • Ensures thresholds prevent false matches")
    print("="*70)
    
    if not os.path.exists(EMBEDDINGS_DIR):
        print("❌ Embeddings directory not found")
        return 1
    
    # Load all users
    users = []
    for d in sorted(os.listdir(EMBEDDINGS_DIR)):
        user_dir = os.path.join(EMBEDDINGS_DIR, d)
        if not os.path.isdir(user_dir):
            continue
        
        parts = d.split('_', 1)
        user_id = parts[0]
        user_name = parts[1] if len(parts) > 1 else d
        
        embs = load_user_embeddings(user_dir)
        if not embs:
            continue
        
        meta_path = os.path.join(user_dir, 'metadata.json')
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        
        users.append({
            'id': user_id,
            'name': user_name,
            'dir': user_dir,
            'embeddings': embs,
            'metadata': meta,
            'old_threshold': meta.get('recommended_threshold', 0.7)
        })
    
    if len(users) == 0:
        print("❌ No users found")
        return 1
    
    print(f"\n📚 Found {len(users)} user(s)")
    print("="*70)
    
    # Recalculate thresholds
    updated_count = 0
    for i, user in enumerate(users):
        print(f"\n👤 User: {user['name']} (ID: {user['id']})")
        print(f"  Old threshold: {user['old_threshold']:.3f}")
        print(f"  Embeddings: {len(user['embeddings'])}")
        
        # Collect all other users' embeddings
        all_other_embs = []
        for other_user in users:
            if other_user['id'] != user['id']:
                all_other_embs.extend(other_user['embeddings'])
        
        # Calculate optimal threshold
        new_threshold = calculate_optimal_threshold(
            user['embeddings'], 
            all_other_embs,
            min_threshold=0.5,
            max_threshold=0.9
        )
        
        # Update metadata
        user['metadata']['recommended_threshold'] = float(new_threshold)
        user['metadata']['threshold_recalculated'] = True
        
        meta_path = os.path.join(user['dir'], 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(user['metadata'], f, indent=2)
        
        print(f"  ✓ Updated threshold: {new_threshold:.3f}")
        if abs(new_threshold - user['old_threshold']) > 0.01:
            updated_count += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Processed {len(users)} user(s)")
    print(f"✓ Updated {updated_count} threshold(s)")
    print("\n💡 Recommendation: Run this script after adding new users")
    print("   to ensure thresholds adapt to prevent false matches")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())