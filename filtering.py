import re
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pep_utils import load_peptide_data_lists, get_peptide_distances
from esm_embeddings import get_esm_model, get_esm_embeddings
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


'''

the type of data i expect:

protein_segment_boundaries = {
    "{protein_id_1}": [[s1, e1], [s2, e2], [s3, e3], [s4, e4]],
    "{protein_id_2}": [[s1, e1], [s2, e2], [s3, e3]]
}

protein_segment_embeddings = {
    "protein_id_1 s1-e1": np.array([...]),  # 1024-dim vector
    "protein_id_1 25-45": np.array([...]),
    # ... etc
}

protein_sequences = {
    "{protein_id_1}": "{protein_id_1_full_sequence}"
    "{protein_id_2}": "{protein_id_2_full_sequence}"
}

# List of reference NES embeddings:
[
    numpy_array_1,  # Known NES peptide 1
    numpy_array_2,  # Known NES peptide 2
    # ... more reference embeddings
]

'''


def filter_segments_by_length(protein_segment_boundaries, min_length=8, max_length=25):
    """
    Filter protein segments by length suitable for NES detection.
    """
    filtered_boundaries = {}

    for protein_id, segments in protein_segment_boundaries.items():
        if segments == "Failed":
            continue

        filtered_segments = []
        for start, end in segments:
            segment_length = end - start + 1
            if min_length <= segment_length <= max_length:
                filtered_segments.append([start, end])

        if filtered_segments:
            filtered_boundaries[protein_id] = filtered_segments
        
    return filtered_boundaries


def extract_segment_sequences(protein_sequences, filtered_boundaries):
    """
    Extract actual amino acid sequences for filtered segments.
    """
    segment_sequences = {}
    
    for protein_id, segments in filtered_boundaries.items():
        if protein_id not in protein_sequences:
            continue
            
        full_sequence = protein_sequences[protein_id]
        
        for start, end in segments:
            segment_key = f"{protein_id} {start}-{end}"
            segment_seq = full_sequence[start:end]
            segment_sequences[segment_key] = segment_seq
    
    return segment_sequences


def score_nes_consensus_pattern(segment_sequences):
    """
    Score segments by multiple NES consensus patterns, return detailed dict:
      { segment_key: { 'score': int, 'matches': [labels...] } }
    """
    # hydrophobic sets
    CORE = set('LIVFM')
    XU = CORE | set('AT')
    STRUCT = CORE | set('AYWP')
    # acidic sets
    ACIDIC = set('DE')

    # pattern descriptors: (name, length, positions, residues, weight)
    forward_patterns = [
        ('Kosugi_1a', 10, [0,4,7,9], [XU,XU,CORE,CORE], 4),
        ('Kosugi_1b', 9,  [0,3,6,8], [XU,XU,CORE,CORE], 3),
        ('Kosugi_1c', 11, [0,4,8,10], [XU,XU,CORE,CORE], 3),
        ('Kosugi_1d', 10, [0,3,7,9], [XU,XU,CORE,CORE], 3),
        ('Kosugi_2', 8,  [0,2,5,7], [XU,XU,CORE,CORE], 3),
        ('Kosugi_3', 10, [0,3,7,9], [XU,XU,CORE,CORE], 2),
    ]

    reverse_patterns = []
    for name, length, pos, res_sets, wt in forward_patterns:
        rev_name = f"{name}_rev"
        rev_pos = [length-1 - p for p in pos]
        reverse_patterns.append((rev_name, length, rev_pos, res_sets, wt//2))

    all_patterns = forward_patterns + reverse_patterns

    results = {}
    for key, seq in segment_sequences.items():
        score = 0
        matches = []

        # sliding-window check
        for name, L, pos_list, res_sets, weight in all_patterns:
            for i in range(len(seq) - L + 1):
                window = seq[i:i+L]
                if all(window[p] in res_sets[j] for j,p in enumerate(pos_list)):
                    score += weight
                    matches.append(f"{name}@{i}")

        # structure-based PKI/Rev-class:
        # PKI: Φ0-X0-3-Φ1-X3-Φ2-X2-3-Φ3-X-Φ4
        for i in range(len(seq)-11):
            for o in range(4):
                indices = [i, i+o+1, i+o+1+4, i+o+1+4+3+1, i+o+1+4+3+1+3]
                if all(idx< len(seq) and seq[idx] in CORE for idx in indices):
                    score += 3
                    matches.append(f"PKI@{i}")
        
        # Rev-class: Φ0-P-X1-Φ2-X2-Φ3-X-Φ4
        for i in range(len(seq)-9):
            w=seq[i:i+9]
            if w[0] in STRUCT and w[1]=='P' and w[3] in CORE and w[6] in CORE and w[8] in CORE:
                score += 3
                matches.append(f"Rev@{i}")

        # hydrophobic ratio 
        hydrophobic_count = sum(1 for aa in seq if aa in CORE)
        ratio = hydrophobic_count / len(seq)
        if ratio >= 0.5:
            score += 2
        elif ratio >= 0.35:
            score += 1

        # proline penalty
        if seq.count('P')>2:
            score -= 1

        # acidic flanks
        if seq[0] in ACIDIC: score +=1
        if seq[-1] in ACIDIC: score +=1

        results[key] = {'score': max(score,0), 'matches': matches}
    return results


def score_segments_with_embeddings(segment_embeddings, positive_reference_embeddings, negative_reference_embeddings):
    """
    Score segments based on similarity to known NES embeddings using the approach from ex4.
    Returns embedding_scores: dict with segment keys and embedding similarity scores
    """
    
    embedding_scores = {}
    
    # if not positive_reference_embeddings:
    #     print("Warning: No reference embeddings provided")
    #     return {key: 0 for key in segment_embeddings.keys()}
    
    # ref_embeddings = np.array(positive_reference_embeddings)
    
    # for segment_key, embedding in segment_embeddings.items():
    #     embedding = embedding.reshape(1, -1)
        
    #     if method == 'distance':
    #         # Calculate mean distance to positive references
    #         distances = cdist(embedding, ref_embeddings, metric='euclidean')[0]
    #         mean_distance = np.mean(distances)

    #         score = 1 / (1 + mean_distance)
        
    #     elif method == 'cosine':
    #         # Calculate cosine similarity to positive references
    #         similarities = cosine_similarity(embedding, ref_embeddings)[0]
    #         score = np.mean(similarities)
        
    #     embedding_scores[segment_key] = score
    
    # return embedding_scores

    segment_keys = list(segment_embeddings.keys())
    segment_embeddings_list = [segment_embeddings[key] for key in segment_keys]
    
    # Calculate distances to positive references
    distances_to_pos = get_peptide_distances(
        segment_embeddings_list, 
        positive_reference_embeddings, 
        reduce_func=np.mean
    )
    
    # Calculate distances to negative references
    distances_to_neg = get_peptide_distances(
        segment_embeddings_list, 
        negative_reference_embeddings, 
        reduce_func=np.mean
    )
        
    # Log-fold difference score (higher = more positive-like)
    scores = np.log1p(distances_to_neg) - np.log1p(distances_to_pos)
        
    
    # Create results dictionary
    for i, key in enumerate(segment_keys):
        embedding_scores[key] = scores[i]
    
    return embedding_scores


def combine_scores_and_rank(consensus_scores, embedding_scores, neural_net_scores, 
                          consensus_weight=0.6, embedding_weight=0.4, neural_net_weight=0.0):
    """
    Combine three scoring methods and rank segments 
    Returns ranked_segments: list of (segment_key, combined_score, details) tuples, sorted by score
    """    
  
    # Get all segment keys
    all_keys = set(consensus_scores.keys()) | set(embedding_scores.keys()) #| set(neural_net_scores.keys())

    if not all_keys:
        return []
    
    # Extract values for normalization
    consensus_values = [consensus_scores.get(key, {}).get('score', 0) for key in all_keys]
    embedding_values = [embedding_scores.get(key, 0) for key in all_keys]
    # neural_values = [neural_net_scores.get(key, 0) for key in all_keys] 
    
    # Min-max normalization
    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [1.0 if v > 0 else 0.0 for v in values]
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    norm_consensus = normalize(consensus_values)
    norm_embedding = normalize(embedding_values)
    # norm_neural = normalize(neural_values) 
    
    # Combine scores
    results = []
    for i, key in enumerate(all_keys):
        combined_score = (consensus_weight * norm_consensus[i] + 
                         embedding_weight * norm_embedding[i])
                        #  + neural_net_weight * norm_neural[i])
        
        details = {
            'combined_score': combined_score,
            'raw_consensus': consensus_scores.get(key, {}).get('score', 0),
            'raw_embedding': embedding_scores.get(key, 0),
            # 'raw_neural_net': neural_net_scores.get(key, 0) if neural_net_scores else 0,
            'pattern_matches': consensus_scores.get(key, {}).get('matches', [])
        }
        
        results.append((key, combined_score, details))
    
    # Sort by combined score (highest first)
    return sorted(results, key=lambda x: x[1], reverse=True)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_project4_pipeline_test(protein_segment_boundaries, protein_segment_embeddings, protein_sequences, positive_reference_embeddings, negative_reference_embeddings):
    """
    Complete test run of Project 4: Precise NES detection using sequence segmentation
    """
    print("="*80)
    print("PROJECT 4: PRECISE NES DETECTION USING SEQUENCE SEGMENTATION")
    print("="*80)
    
    print("\n STEP 2: Filtering segments by length (8-15 amino acids)")
    
    filtered_boundaries = filter_segments_by_length(
        protein_segment_boundaries, 
        min_length=8, 
        max_length=25
    )
    
    total_segments_before = sum(len(segments) for segments in protein_segment_boundaries.values())
    total_segments_after = sum(len(segments) for segments in filtered_boundaries.values())
    
    print(f" Segments before filtering: {total_segments_before}")
    print(f" Segments after filtering: {total_segments_after}")
    print(f" Filtering efficiency: {total_segments_after/total_segments_before*100:.1f}%")
    
    # # Show filtered segments details
    # for protein_id, segments in filtered_boundaries.items():
    #     print(f"  - {protein_id}: {len(segments)} segments")
    #     for start, end in segments:
    #         length = end - start + 1
    #         print(f"    • {start}-{end} (length: {length})")
    
    
    print("\n STEP 3: Extracting segment sequences")

    segment_sequences = extract_segment_sequences(protein_sequences, filtered_boundaries)
    
    print(f" Extracted {len(segment_sequences)} segment sequences")
 

    print("\n STEP 4: Scoring segments with NES consensus patterns")
    
    consensus_results = score_nes_consensus_pattern(segment_sequences)
    
    # Extract scores for analysis
    consensus_scores = {key: result['score'] for key, result in consensus_results.items()}
    
    print(f" Scored {len(consensus_results)} segments")
    print(f" Score range: {min(consensus_scores.values()):.1f} - {max(consensus_scores.values()):.1f}")
    print(f" Average score: {np.mean(list(consensus_scores.values())):.2f}")
    
    # Show top 5 consensus matches
    top_consensus = sorted(consensus_results.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
    print("\nTop 5 consensus pattern matches:")
    for i, (segment_key, result) in enumerate(top_consensus):
        sequence = segment_sequences[segment_key]
        print(f"  {i+1}. {segment_key}")
        print(f"     Sequence: {sequence}")
        print(f"     Score: {result['score']}")
        print(f"     Patterns: {', '.join(result['matches'][:])}")
        # print(f"     Patterns: {', '.join(result['matches'][:3])}{'...' if len(result['matches']) > 3 else ''}")
    
   
    print("\n STEP 5: Scoring segments with embedding similarity")
    
    # Filter embeddings to only include segments that passed length filtering
    filtered_embeddings = {
        key: emb for key, emb in protein_segment_embeddings.items() 
        if key in segment_sequences
    }
    
    embedding_scores = score_segments_with_embeddings(
        filtered_embeddings, 
        positive_reference_embeddings,
        negative_reference_embeddings
    )
    
    print(f" Scored {len(embedding_scores)} segments with embedding similarity")
    print(f" Score range: {min(embedding_scores.values()):.3f} - {max(embedding_scores.values()):.3f}")
    print(f" Average score: {np.mean(list(embedding_scores.values())):.3f}")
    
    # Show top 5 embedding matches
    top_embedding = sorted(embedding_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 embedding similarity matches:")
    for i, (segment_key, score) in enumerate(top_embedding):
        sequence = segment_sequences[segment_key]
        print(f"  {i+1}. {segment_key}")
        print(f"     Sequence: {sequence}")
        print(f"     Embedding score: {score:.4f}")
 

    print("\n STEP 6: Combining scores and ranking segments...")
    
    ranked_segments = combine_scores_and_rank(
        consensus_results, 
        embedding_scores, 
        neural_net_scores=None,  # No neural network for this test
        consensus_weight=0.6,
        embedding_weight=0.4
    )
    
    print(f" Ranked {len(ranked_segments)} segments")
    print(f" Weighting: 60% consensus patterns, 40% embedding similarity")
    

    print("\n STEP 7: TOP NES CANDIDATES")
    print("="*80)
    
    top_n = min(10, len(ranked_segments))
    
    for i, (segment_key, combined_score, details) in enumerate(ranked_segments[:top_n]):
        sequence = segment_sequences[segment_key]
        print(f"\nRANK #{i+1}: {segment_key}")
        print(f"Sequence: {sequence}")
        print(f"Length: {len(sequence)} amino acids")
        print(f"Combined Score: {combined_score:.4f}")
        print(f"  • Consensus Score: {details['raw_consensus']} (patterns: {len(details['pattern_matches'])})")
        print(f"  • Embedding Score: {details['raw_embedding']:.4f}")
        print(f"  • Pattern Matches: {', '.join(details['pattern_matches'][:3])}{'...' if len(details['pattern_matches']) > 3 else ''}")
        print("-" * 60)
    
#     # ========================================================================
#     # STEP 8: GENERATE SUMMARY STATISTICS
#     # ========================================================================
#     print("\n📊 STEP 8: SUMMARY STATISTICS")
#     print("="*50)
    
#     # Score distributions
#     all_combined_scores = [score for _, score, _ in ranked_segments]
#     all_consensus_scores = [details['raw_consensus'] for _, _, details in ranked_segments]
#     all_embedding_scores = [details['raw_embedding'] for _, _, details in ranked_segments]
    
#     print(f"Combined Scores:")
#     print(f"  Range: {min(all_combined_scores):.4f} - {max(all_combined_scores):.4f}")
#     print(f"  Mean: {np.mean(all_combined_scores):.4f}")
#     print(f"  Std: {np.std(all_combined_scores):.4f}")
    
#     print(f"\nConsensus Scores:")
#     print(f"  Range: {min(all_consensus_scores)} - {max(all_consensus_scores)}")
#     print(f"  Mean: {np.mean(all_consensus_scores):.2f}")
    
#     print(f"\nEmbedding Scores:")
#     print(f"  Range: {min(all_embedding_scores):.4f} - {max(all_embedding_scores):.4f}")
#     print(f"  Mean: {np.mean(all_embedding_scores):.4f}")
    
#     # Pattern match analysis
#     all_pattern_matches = [match for _, _, details in ranked_segments for match in details['pattern_matches']]
#     pattern_counts = {}
#     for match in all_pattern_matches:
#         pattern_type = match.split('@')[0]
#         pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
#     print(f"\nPattern Match Distribution:")
#     for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
#         print(f"  {pattern}: {count} matches")
    
#     # ========================================================================
#     # STEP 9: SAVE RESULTS
#     # ========================================================================
#     print("\n💾 STEP 9: Saving results...")
    
#     # Prepare results for CSV
#     results_data = []
#     for i, (segment_key, combined_score, details) in enumerate(ranked_segments):
#         protein_id = segment_key.split()[0]
#         positions = segment_key.split()[1]
#         sequence = segment_sequences[segment_key]
        
#         results_data.append({
#             'rank': i + 1,
#             'segment_id': segment_key,
#             'protein_id': protein_id,
#             'positions': positions,
#             'sequence': sequence,
#             'sequence_length': len(sequence),
#             'combined_score': combined_score,
#             'consensus_score': details['raw_consensus'],
#             'embedding_score': details['raw_embedding'],
#             'num_pattern_matches': len(details['pattern_matches']),
#             'pattern_matches': '; '.join(details['pattern_matches'])
#         })
    
#     # Save to CSV
#     results_df = pd.DataFrame(results_data)
#     results_df.to_csv('project4_nes_candidates.csv', index=False)
#     print(f"✓ Results saved to 'project4_nes_candidates.csv'")
    
#     # ========================================================================
#     # STEP 10: GENERATE PLOTS
#     # ========================================================================
#     print("\n📈 STEP 10: Generating summary plots...")
    
#     generate_summary_plots(results_df)
#     print("✓ Summary plots saved as 'project4_summary_plots.png'")
    
#     print("\n" + "="*80)
#     print("✅ PROJECT 4 PIPELINE COMPLETED SUCCESSFULLY!")
#     print("="*80)
#     print(f"🎯 Top NES candidate: {ranked_segments[0][0]}")
#     print(f"🧬 Sequence: {segment_sequences[ranked_segments[0][0]]}")
#     print(f"⭐ Combined score: {ranked_segments[0][1]:.4f}")
#     print(f"📁 Results saved to: project4_nes_candidates.csv")
#     print(f"📊 Plots saved to: project4_summary_plots.png")
    
#     return {
#         'ranked_segments': ranked_segments,
#         'segment_sequences': segment_sequences,
#         'consensus_results': consensus_results,
#         'embedding_scores': embedding_scores,
#         'results_dataframe': results_df
#     }

# def generate_summary_plots(results_df):
#     """Generate summary plots for Project 4 results"""
    
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
#     # Plot 1: Combined score distribution
#     axes[0, 0].hist(results_df['combined_score'], bins=20, alpha=0.7, color='blue', edgecolor='black')
#     axes[0, 0].set_xlabel('Combined Score')
#     axes[0, 0].set_ylabel('Number of Segments')
#     axes[0, 0].set_title('Distribution of Combined Scores')
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # Plot 2: Consensus vs Embedding scores
#     scatter = axes[0, 1].scatter(results_df['consensus_score'], results_df['embedding_score'], 
#                                 alpha=0.6, c=results_df['combined_score'], cmap='viridis')
#     axes[0, 1].set_xlabel('Consensus Pattern Score')
#     axes[0, 1].set_ylabel('Embedding Similarity Score')
#     axes[0, 1].set_title('Consensus vs Embedding Scores')
#     axes[0, 1].grid(True, alpha=0.3)
#     plt.colorbar(scatter, ax=axes[0, 1], label='Combined Score')
    
#     # Plot 3: Sequence length distribution
#     axes[1, 0].hist(results_df['sequence_length'], bins=range(8, 17), alpha=0.7, 
#                    color='green', edgecolor='black')
#     axes[1, 0].set_xlabel('Sequence Length (amino acids)')
#     axes[1, 0].set_ylabel('Number of Segments')
#     axes[1, 0].set_title('Distribution of Segment Lengths')
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # Plot 4: Top 15 candidates
#     top_15 = results_df.head(15)
#     bars = axes[1, 1].bar(range(1, len(top_15) + 1), top_15['combined_score'], 
#                          alpha=0.7, color='red', edgecolor='black')
#     axes[1, 1].set_xlabel('Rank')
#     axes[1, 1].set_ylabel('Combined Score')
#     axes[1, 1].set_title('Top 15 NES Candidates')
#     axes[1, 1].grid(True, alpha=0.3)
    
#     # Add value labels on bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
#                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig('project4_summary_plots.png', dpi=300, bbox_inches='tight')
#     plt.close()








def load_zps_segments_for_evaluation(file_path):
    """
    Load segments from JSONL file and prepare data for your pipeline.
    
    Returns:
    Dict with data needed for evaluation using your existing functions
    """
    segments = []
    with open(file_path, 'r') as f:
        for line in f:
            segments.append(json.loads(line))
    
    # Prepare data structures matching your code format
    protein_segment_boundaries = {}
    protein_segment_embeddings = {}
    protein_sequences = {}
    true_labels = {}
    
    # Group segments by protein ID
    proteins = {}
    for segment in segments:
        protein_id = segment['id']
        if protein_id not in proteins:
            proteins[protein_id] = []
        proteins[protein_id].append(segment)
    
    # Create data structures
    for protein_id, protein_segments in proteins.items():
        boundaries = []
        
        for segment in protein_segments:
            start, end = segment['coords']
            segment_key = f"{protein_id} {start}-{end}"
            
            # Store boundary
            boundaries.append([start, end])
            
            # Store embedding
            protein_segment_embeddings[segment_key] = np.array(segment['embedding'])
            
            # Store true label
            true_labels[segment_key] = segment['label']
        
        # Store boundaries for this protein
        protein_segment_boundaries[protein_id] = boundaries
        
        # You'll need to provide protein sequences separately or extract from somewhere
        # For now, create placeholder - replace with actual sequence loading
        protein_sequences[protein_id] = "A" * 1000  # Placeholder
    
    return {
        'protein_segment_boundaries': protein_segment_boundaries,
        'protein_segment_embeddings': protein_segment_embeddings,
        'protein_sequences': protein_sequences,
        'true_labels': true_labels,
        'raw_segments': segments
    }

def evaluate_project4_accuracy(jsonl_file_path, 
                              positive_reference_embeddings,
                              negative_reference_embeddings,
                              protein_sequences_dict=None,
                              min_length=8, max_length=25,
                              consensus_weight=0.6, embedding_weight=0.4):
    """
    Evaluate accuracy of your Project 4 scoring functions.
    
    Parameters:
    jsonl_file_path: path to zps_segments.jsonl
    positive_reference_embeddings: your positive reference embeddings
    negative_reference_embeddings: your negative reference embeddings
    protein_sequences_dict: dict with actual protein sequences (optional)
    min_length, max_length: length filters for segments
    consensus_weight, embedding_weight: scoring weights
    
    Returns:
    Dict with evaluation results
    """
    print("="*60)
    print("PROJECT 4 ACCURACY EVALUATION")
    print("="*60)
    
    # Load test data
    print("Loading test data from JSONL...")
    eval_data = load_zps_segments_for_evaluation(jsonl_file_path)
    
    # Use provided protein sequences if available
    if protein_sequences_dict:
        eval_data['protein_sequences'].update(protein_sequences_dict)
    
    # Get true labels
    true_labels_dict = eval_data['true_labels']
    
    # Filter segments by length (using your function)
    print("Filtering segments by length...")
    filtered_boundaries = filter_segments_by_length(
        eval_data['protein_segment_boundaries'], 
        min_length=min_length, 
        max_length=max_length
    )
    
    # Extract segment sequences (using your function)
    print("Extracting segment sequences...")
    segment_sequences = extract_segment_sequences(
        eval_data['protein_sequences'], 
        filtered_boundaries
    )
    
    # Get filtered embeddings and true labels
    filtered_embeddings = {
        key: emb for key, emb in eval_data['protein_segment_embeddings'].items() 
        if key in segment_sequences
    }
    
    filtered_true_labels = {
        key: label for key, label in true_labels_dict.items()
        if key in segment_sequences
    }
    
    if not filtered_true_labels:
        print("ERROR: No segments found after filtering!")
        return None
    
    print(f"Evaluating {len(filtered_true_labels)} segments...")
    
    # Prepare arrays for evaluation
    segment_keys = list(filtered_true_labels.keys())
    true_labels = np.array([filtered_true_labels[key] for key in segment_keys])
    
    results = {}
    
    # 1. EVALUATE CONSENSUS PATTERN SCORING
    print("\n1. Evaluating consensus pattern scoring...")
    consensus_results = score_nes_consensus_pattern(segment_sequences)
    consensus_scores = np.array([
        consensus_results.get(key, {}).get('score', 0) for key in segment_keys
    ])
    
    # Convert to binary predictions using median threshold
    consensus_threshold = np.median(consensus_scores) if len(consensus_scores) > 0 else 0
    consensus_predictions = (consensus_scores >= consensus_threshold).astype(int)
    
    results['consensus'] = {
        'scores': consensus_scores,
        'predictions': consensus_predictions,
        'threshold': consensus_threshold,
        'accuracy': accuracy_score(true_labels, consensus_predictions),
        'precision': precision_score(true_labels, consensus_predictions, zero_division=0),
        'recall': recall_score(true_labels, consensus_predictions, zero_division=0),
        'f1': f1_score(true_labels, consensus_predictions, zero_division=0),
        'auc': roc_auc_score(true_labels, consensus_scores) if len(np.unique(true_labels)) > 1 else 0
    }
    
    # 2. EVALUATE EMBEDDING SIMILARITY SCORING
    print("2. Evaluating embedding similarity scoring...")
    embedding_scores_dict = score_segments_with_embeddings(
        filtered_embeddings, 
        positive_reference_embeddings,
        negative_reference_embeddings
    )
    
    embedding_scores = np.array([
        embedding_scores_dict.get(key, 0) for key in segment_keys
    ])
    
    # Convert to binary predictions (positive scores = positive class)
    embedding_predictions = (embedding_scores > 0).astype(int)
    
    results['embedding'] = {
        'scores': embedding_scores,
        'predictions': embedding_predictions,
        'threshold': 0.0,
        'accuracy': accuracy_score(true_labels, embedding_predictions),
        'precision': precision_score(true_labels, embedding_predictions, zero_division=0),
        'recall': recall_score(true_labels, embedding_predictions, zero_division=0),
        'f1': f1_score(true_labels, embedding_predictions, zero_division=0),
        'auc': roc_auc_score(true_labels, embedding_scores) if len(np.unique(true_labels)) > 1 else 0
    }
    
    # 3. EVALUATE COMBINED SCORING
    print("3. Evaluating combined scoring...")
    ranked_segments = combine_scores_and_rank(
        consensus_results, 
        embedding_scores_dict, 
        neural_net_scores=None,
        consensus_weight=consensus_weight,
        embedding_weight=embedding_weight
    )
    
    # Extract combined scores in same order as segment_keys
    combined_scores = np.array([
        next(details['combined_score'] for seg_key, _, details in ranked_segments if seg_key == key)
        for key in segment_keys
    ])
    
    # Convert to binary predictions using median threshold
    combined_threshold = np.median(combined_scores)
    combined_predictions = (combined_scores >= combined_threshold).astype(int)
    
    results['combined'] = {
        'scores': combined_scores,
        'predictions': combined_predictions,
        'threshold': combined_threshold,
        'accuracy': accuracy_score(true_labels, combined_predictions),
        'precision': precision_score(true_labels, combined_predictions, zero_division=0),
        'recall': recall_score(true_labels, combined_predictions, zero_division=0),
        'f1': f1_score(true_labels, combined_predictions, zero_division=0),
        'auc': roc_auc_score(true_labels, combined_scores) if len(np.unique(true_labels)) > 1 else 0
    }
    
    # Store additional info
    results['true_labels'] = true_labels
    results['segment_keys'] = segment_keys
    results['evaluation_params'] = {
        'min_length': min_length,
        'max_length': max_length,
        'consensus_weight': consensus_weight,
        'embedding_weight': embedding_weight,
        'n_segments': len(segment_keys),
        'n_positive': np.sum(true_labels),
        'n_negative': len(true_labels) - np.sum(true_labels)
    }
    
    return results

def print_accuracy_results(results):
    """Print formatted accuracy evaluation results."""
    if not results:
        print("No results to display!")
        return
        
    print("\n" + "="*60)
    print("ACCURACY EVALUATION RESULTS")
    print("="*60)
    
    params = results['evaluation_params']
    print(f"Dataset: {params['n_segments']} segments ({params['n_positive']} positive, {params['n_negative']} negative)")
    print(f"Length filter: {params['min_length']}-{params['max_length']} amino acids")
    print(f"Weights: {params['consensus_weight']:.1f} consensus + {params['embedding_weight']:.1f} embedding")
    
    methods = ['consensus', 'embedding', 'combined']
    
    for method in methods:
        if method not in results:
            continue
            
        metrics = results[method]
        print(f"\n{method.upper()} SCORING:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['auc']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.4f}")

# def plot_accuracy_results(results, save_path="project4_accuracy_evaluation.png"):
#     """Generate accuracy evaluation plots."""
#     if not results:
#         return
        
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
#     methods = [k for k in ['consensus', 'embedding', 'combined'] if k in results]
#     true_labels = results['true_labels']
    
#     # ROC Curves
#     axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
#     for method in methods:
#         if 'auc' in results[method] and results[method]['auc'] > 0:
#             fpr, tpr, _ = roc_curve(true_labels, results[method]['scores'])
#             auc_score = results[method]['auc']
#             axes[0, 0].plot(fpr, tpr, label=f'{method.title()} (AUC = {auc_score:.3f})')
    
#     axes[0, 0].set_xlabel('False Positive Rate')
#     axes[0, 0].set_ylabel('True Positive Rate')
#     axes[0, 0].set_title('ROC Curves')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # Accuracy Comparison
#     accuracies = [results[method]['accuracy'] for method in methods]
#     axes[0, 1].bar(methods, accuracies, alpha=0.7, color=['blue', 'green', 'red'])
#     axes[0, 1].set_ylabel('Accuracy')
#     axes[0, 1].set_title('Accuracy Comparison')
#     axes[0, 1].set_ylim(0, 1)
    
#     # F1-Score Comparison
#     f1_scores = [results[method]['f1'] for method in methods]
#     axes[1, 0].bar(methods, f1_scores, alpha=0.7, color=['blue', 'green', 'red'])
#     axes[1, 0].set_ylabel('F1-Score')
#     axes[1, 0].set_title('F1-Score Comparison')
#     axes[1, 0].set_ylim(0, 1)
    
#     # Confusion Matrix for best method
#     best_method = max(methods, key=lambda m: results[m]['accuracy'])
#     cm = confusion_matrix(true_labels, results[best_method]['predictions'])
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
#     axes[1, 1].set_title(f'Confusion Matrix - {best_method.title()}')
#     axes[1, 1].set_xlabel('Predicted')
#     axes[1, 1].set_ylabel('True')
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()

# Example usage function that integrates with your existing code
def run_accuracy_evaluation_with_your_data():
    """
    Example of how to run accuracy evaluation with your existing data.
    """
    # Load your reference embeddings (as you already do)
    pos_pep, neg_pep, doubt_list = load_peptide_data_lists()
    model, alphabet, batch_converter, device = get_esm_model(embedding_size=640)
    
    positive_reference_embeddings = get_esm_embeddings(
        pos_pep, model, alphabet, batch_converter, device,
        embedding_layer=17, sequence_embedding=True
    )
    
    negative_reference_embeddings = get_esm_embeddings(
        neg_pep, model, alphabet, batch_converter, device,
        embedding_layer=17, sequence_embedding=True
    )
    
    # Load your protein sequences if you have them
    with open("data/protein_sequences.pkl", "rb") as f:
        protein_sequences = pickle.load(f)
    
    # Run accuracy evaluation
    results = evaluate_project4_accuracy(
        jsonl_file_path="zps_segments.jsonl",
        positive_reference_embeddings=positive_reference_embeddings,
        negative_reference_embeddings=negative_reference_embeddings,
        protein_sequences_dict=protein_sequences,
        min_length=8,
        max_length=25,
        consensus_weight=0.6,
        embedding_weight=0.4
    )
    
    # Print and plot results
    print_accuracy_results(results)
    # plot_accuracy_results(results)
    
    return results



if __name__ == "__main__":
    with open("data/protein_segments.pkl", "rb") as f:
        protein_segment_boundaries = pickle.load(f)
    
    with open("data/protein_segment_embeddings.pkl", "rb") as f:
        protein_segment_embeddings = pickle.load(f)
    
    with open("data/protein_sequences.pkl", "rb") as f:
        protein_sequences = pickle.load(f)
    
    pos_pep, neg_pep, doubt_list = load_peptide_data_lists()
    model, alphabet, batch_converter, device = get_esm_model(embedding_size=640)
    positive_reference_embeddings = get_esm_embeddings(pos_pep, model, alphabet, batch_converter, device,
                                                      embedding_layer=17, sequence_embedding=True)
                        
    negative_reference_embeddings = get_esm_embeddings(neg_pep, model, alphabet, batch_converter, device,
                                                      embedding_layer=17, sequence_embedding=True)
    
    results = run_project4_pipeline_test(protein_segment_boundaries, protein_segment_embeddings, 
                                         protein_sequences, positive_reference_embeddings, negative_reference_embeddings)
    
 
    # Add accuracy evaluation
    print("\n" + "="*80)
    print("RUNNING ACCURACY EVALUATION")
    print("="*80)
    
    accuracy_results = run_accuracy_evaluation_with_your_data()
