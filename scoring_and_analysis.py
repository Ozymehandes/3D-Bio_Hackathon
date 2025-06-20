import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

from pep_utils import load_peptide_data_lists, get_peptide_distances
from esm_embeddings import get_esm_model, get_esm_embeddings
from plot import plot_roc_curve  # existing ROC plot function



def filter_segments_by_length(protein_segment_boundaries, min_length=8, max_length=25):
    """Filter protein segments by length suitable for NES detection."""
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
    """Extract actual amino acid sequences for filtered segments."""
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
    """Score segments by multiple NES consensus patterns."""
    CORE = set('LIVFM')
    XU = CORE | set('AT')
    STRUCT = CORE | set('AYWP')
    ACIDIC = set('DE')

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

        for name, L, pos_list, res_sets, weight in all_patterns:
            for i in range(len(seq) - L + 1):
                window = seq[i:i+L]
                if all(window[p] in res_sets[j] for j,p in enumerate(pos_list)):
                    score += weight
                    matches.append(f"{name}@{i}")

        for i in range(len(seq)-11):
            for o in range(4):
                indices = [i, i+o+1, i+o+1+4, i+o+1+4+3+1, i+o+1+4+3+1+3]
                if all(idx< len(seq) and seq[idx] in CORE for idx in indices):
                    score += 3
                    matches.append(f"PKI@{i}")
        
        for i in range(len(seq)-9):
            w=seq[i:i+9]
            if w[0] in STRUCT and w[1]=='P' and w[3] in CORE and w[6] in CORE and w[8] in CORE:
                score += 3
                matches.append(f"Rev@{i}")

        hydrophobic_count = sum(1 for aa in seq if aa in CORE)
        ratio = hydrophobic_count / len(seq)
        if ratio >= 0.5:
            score += 2
        elif ratio >= 0.35:
            score += 1

        if seq.count('P')>2:
            score -= 1

        if seq[0] in ACIDIC: score +=1
        if seq[-1] in ACIDIC: score +=1

        results[key] = {'score': max(score,0), 'matches': matches}
    return results



def score_segments_with_embeddings(segment_embeddings, positive_reference_embeddings, negative_reference_embeddings):
    """Score segments using log-fold difference approach."""
    embedding_scores = {}
    
    segment_keys = list(segment_embeddings.keys())
    segment_embeddings_list = [segment_embeddings[key] for key in segment_keys]
    
    distances_to_pos = get_peptide_distances(
        segment_embeddings_list, positive_reference_embeddings, reduce_func=np.mean
    )
    distances_to_neg = get_peptide_distances(
        segment_embeddings_list, negative_reference_embeddings, reduce_func=np.mean
    )
    
    scores = np.log1p(distances_to_neg) - np.log1p(distances_to_pos)
    
    for i, key in enumerate(segment_keys):
        embedding_scores[key] = scores[i]
    
    return embedding_scores



def combine_scores_and_rank(consensus_scores, embedding_scores, neural_net_scores, 
                          consensus_weight=0.4, embedding_weight=0.4, neural_net_weight=0.3, union_keys=True):
    """Combine scoring methods and rank segments."""
    if union_keys:
        all_keys = set(consensus_scores.keys()) | set(embedding_scores.keys()) | set(neural_net_scores.keys())
    else:
        all_keys = set(consensus_scores.keys()) & set(embedding_scores.keys()) & set(neural_net_scores.keys())

    if not all_keys:
        return []
    
    consensus_values = [consensus_scores.get(key, {}).get('score', 0) for key in all_keys]
    embedding_values = [embedding_scores.get(key, 0) for key in all_keys]
    neural_values = [neural_net_scores.get(key, 0) for key in all_keys] 

    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [1.0 if v > 0 else 0.0 for v in values]
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    norm_consensus = normalize(consensus_values)
    norm_embedding = normalize(embedding_values)
    norm_neural = normalize(neural_values) 
    
    results = []
    for i, key in enumerate(all_keys):
        combined_score = (consensus_weight * norm_consensus[i] + 
                         embedding_weight * norm_embedding[i] +
                         neural_net_weight * norm_neural[i])
        
        details = {
            'combined_score': combined_score,
            'raw_consensus': consensus_scores.get(key, {}).get('score', 0),
            'raw_embedding': embedding_scores.get(key, 0),
            'raw_neural_net': neural_net_scores.get(key, 0),
            'pattern_matches': consensus_scores.get(key, {}).get('matches', [])
        }
        
        results.append((key, combined_score, details))
    
    return sorted(results, key=lambda x: x[1], reverse=True)



# Load true labels for accuracy evaluation
def load_true_labels(jsonl_file_path):
    """Load true labels from JSONL file."""
    true_labels_dict = {}
    try:
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                segment = json.loads(line)
                start, end = segment['coords']
                segment_key = f"{segment['id']} {start}-{end}"
                true_labels_dict[segment_key] = segment['label']
        print(f"Loaded true labels for {len(true_labels_dict)} segments")
    except FileNotFoundError:
        print("Warning: True labels file not found. Accuracy evaluation will be skipped.")
    except Exception as e:
        print(f"Warning: Error loading true labels: {e}")
    return true_labels_dict



def find_optimal_threshold(true_labels, predicted_scores, metric='f1', n_thresholds=100):
    """
    Find the optimal threshold for a given metric.
    Returns:
    best_threshold: optimal threshold
    best_score: best metric value
    all_thresholds: array of all tested thresholds
    all_scores: array of all metric values
    """
    
    # Create threshold range
    min_score = np.min(predicted_scores)
    max_score = np.max(predicted_scores)
    thresholds = np.linspace(min_score, max_score, n_thresholds)
    
    scores = []
    
    for threshold in thresholds:
        predictions = (predicted_scores >= threshold).astype(int)
        
        if metric == 'accuracy':
            score = accuracy_score(true_labels, predictions)
        elif metric == 'precision':
            score = precision_score(true_labels, predictions, zero_division=0)
        elif metric == 'recall':
            score = recall_score(true_labels, predictions, zero_division=0)
        elif metric == 'f1':
            score = f1_score(true_labels, predictions, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return best_threshold, best_score, thresholds, scores



def plot_threshold_analysis(true_labels, predicted_scores, method_name, save_path=None):
    """
    Plot accuracy, precision, recall, and F1 vs threshold for a scoring method.
    
    Returns:
    fig, axes: matplotlib figure and axes
    optimal_thresholds: dict with optimal thresholds for each metric
    """
    
    # Find optimal thresholds for different metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    optimal_thresholds = {}
    all_results = {}
    
    for metric in metrics:
        best_thresh, best_score, thresholds, scores = find_optimal_threshold(
            true_labels, predicted_scores, metric=metric
        )
        optimal_thresholds[metric] = {
            'threshold': best_thresh,
            'score': best_score
        }
        all_results[metric] = {
            'thresholds': thresholds,
            'scores': scores
        }
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Threshold Analysis - {method_name}', fontsize=16)
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot 1: All metrics vs threshold
    ax1 = axes[0, 0]
    for i, metric in enumerate(metrics):
        thresholds = all_results[metric]['thresholds']
        scores = all_results[metric]['scores']
        best_thresh = optimal_thresholds[metric]['threshold']
        best_score = optimal_thresholds[metric]['score']
        
        ax1.plot(thresholds, scores, label=f'{metric.title()}', color=colors[i])
        ax1.axvline(best_thresh, color=colors[i], linestyle='--', alpha=0.7)
        ax1.plot(best_thresh, best_score, 'o', color=colors[i], markersize=8)
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('All Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy detailed view
    ax2 = axes[0, 1]
    acc_thresholds = all_results['accuracy']['thresholds']
    acc_scores = all_results['accuracy']['scores']
    best_acc_thresh = optimal_thresholds['accuracy']['threshold']
    best_acc_score = optimal_thresholds['accuracy']['score']
    
    ax2.plot(acc_thresholds, acc_scores, 'b-', linewidth=2, label='Accuracy')
    ax2.axvline(best_acc_thresh, color='red', linestyle='--', 
                label=f'Best Threshold = {best_acc_thresh:.3f}')
    ax2.plot(best_acc_thresh, best_acc_score, 'ro', markersize=10, 
             label=f'Best Accuracy = {best_acc_score:.3f}')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Threshold (Detailed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: F1 detailed view
    ax3 = axes[1, 0]
    f1_thresholds = all_results['f1']['thresholds']
    f1_scores = all_results['f1']['scores']
    best_f1_thresh = optimal_thresholds['f1']['threshold']
    best_f1_score = optimal_thresholds['f1']['score']
    
    ax3.plot(f1_thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
    ax3.axvline(best_f1_thresh, color='red', linestyle='--', 
                label=f'Best Threshold = {best_f1_thresh:.3f}')
    ax3.plot(best_f1_thresh, best_f1_score, 'ro', markersize=10, 
             label=f'Best F1 = {best_f1_score:.3f}')
    
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score vs Threshold (Detailed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary of optimal thresholds
    ax4 = axes[1, 1]
    metric_names = list(optimal_thresholds.keys())
    threshold_values = [optimal_thresholds[m]['threshold'] for m in metric_names]
    score_values = [optimal_thresholds[m]['score'] for m in metric_names]
    
    bars = ax4.bar(metric_names, score_values, color=colors[:len(metric_names)], alpha=0.7)
    ax4.set_ylabel('Best Score Value')
    ax4.set_title('Optimal Scores by Metric')
    ax4.set_ylim(0, 1)
    
    # Add threshold values on top of bars
    for i, (bar, thresh) in enumerate(zip(bars, threshold_values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'T={thresh:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    else:
        plt.show()
    
    return fig, axes, optimal_thresholds



# Accuracy evaluation function
def evaluate_accuracy(segment_sequences, consensus_scores, embedding_scores, neural_scores, ranked_segments, true_labels_dict, optimization_metric='f1', union_keys=True):
    """Evaluate accuracy of all scoring methods."""
    if not true_labels_dict:
        print("No true labels available for accuracy evaluation.")
        return None
    if union_keys:
    # Filter to segments that have true labels
        common_keys = set(segment_sequences.keys()) & set(true_labels_dict.keys())
        if not common_keys:
            print("No common segments found between predictions and true labels.")
            return None
    else:
        consensus_keys = set(consensus_scores.keys())
        embedding_keys = set(embedding_scores.keys())
        neural_keys = set(neural_scores.keys())
        true_label_keys = set(true_labels_dict.keys())
        
        # Use intersection to get only segments scored by all methods AND have true labels
        common_keys = consensus_keys & embedding_keys & neural_keys & true_label_keys
        
        if not common_keys:
            print("No common segments found between all scoring methods and true labels.")
            print(f"  Consensus segments: {len(consensus_keys)}")
            print(f"  Embedding segments: {len(embedding_keys)}")
            print(f"  Neural segments: {len(neural_keys)}")
            print(f"  True label segments: {len(true_label_keys)}")
            return None
    
    print(f"\nEvaluating accuracy on {len(common_keys)} segments with ground truth labels...")
    print(f"Optimizing thresholds for: {optimization_metric}")
    
    segment_keys = list(common_keys)
    true_labels = np.array([true_labels_dict[key] for key in segment_keys])
    
    results = {}
    
    # 1. CONSENSUS SCORING ACCURACY WITH OPTIMAL THRESHOLD
    print("\n1. Finding optimal threshold for consensus scoring...")
    consensus_scores_arr = np.array([
        consensus_scores.get(key, {}).get('score', 0) for key in segment_keys
    ])
    
    # Find optimal threshold
    best_consensus_thresh, best_consensus_score, _, _ = find_optimal_threshold(
        true_labels, consensus_scores_arr, metric=optimization_metric
    )
    
    # Plot threshold analysis for consensus
    fig_consensus, _, consensus_optimal_thresholds = plot_threshold_analysis(
        true_labels, consensus_scores_arr, 'Consensus Pattern Scoring',
        save_path='consensus_threshold_analysis.png'
    )
    
    # Calculate metrics with optimal threshold
    consensus_pred = (consensus_scores_arr >= best_consensus_thresh).astype(int)
    
    results['consensus'] = {
        'optimal_threshold': best_consensus_thresh,
        'optimal_metric_value': best_consensus_score,
        'all_optimal_thresholds': consensus_optimal_thresholds,
        'accuracy': accuracy_score(true_labels, consensus_pred),
        'precision': precision_score(true_labels, consensus_pred, zero_division=0),
        'recall': recall_score(true_labels, consensus_pred, zero_division=0),
        'f1': f1_score(true_labels, consensus_pred, zero_division=0),
        'auc': roc_auc_score(true_labels, consensus_scores_arr) if len(np.unique(true_labels)) > 1 else 0,
        'scores': consensus_scores_arr,
        'predictions': consensus_pred
    }
    
    # 2. EMBEDDING SCORING ACCURACY WITH OPTIMAL THRESHOLD
    print("\n2. Finding optimal threshold for embedding scoring...")
    embedding_scores_arr = np.array([
        embedding_scores.get(key, 0) for key in segment_keys
    ])
    
    # Find optimal threshold
    best_embedding_thresh, best_embedding_score, _, _ = find_optimal_threshold(
        true_labels, embedding_scores_arr, metric=optimization_metric
    )
    
    # Plot threshold analysis for embedding
    fig_embedding, _, embedding_optimal_thresholds = plot_threshold_analysis(
        true_labels, embedding_scores_arr, 'Embedding Similarity Scoring',
        save_path='embedding_threshold_analysis.png'
    )
    
    # Calculate metrics with optimal threshold
    embedding_pred = (embedding_scores_arr >= best_embedding_thresh).astype(int)
    
    results['embedding'] = {
        'optimal_threshold': best_embedding_thresh,
        'optimal_metric_value': best_embedding_score,
        'all_optimal_thresholds': embedding_optimal_thresholds,
        'accuracy': accuracy_score(true_labels, embedding_pred),
        'precision': precision_score(true_labels, embedding_pred, zero_division=0),
        'recall': recall_score(true_labels, embedding_pred, zero_division=0),
        'f1': f1_score(true_labels, embedding_pred, zero_division=0),
        'auc': roc_auc_score(true_labels, embedding_scores_arr) if len(np.unique(true_labels)) > 1 else 0,
        'scores': embedding_scores_arr,
        'predictions': embedding_pred
    }

     # 3. NEURAL SCORING ACCURACY WITH OPTIMAL THRESHOLD
    print("\n2. Finding optimal threshold for neural scoring...")
    neural_scores_arr = np.array([
        neural_scores.get(key, 0) for key in segment_keys
    ])
    
    # Find optimal threshold
    best_neural_thresh, best_neural_score, _, _ = find_optimal_threshold(
        true_labels, neural_scores_arr, metric=optimization_metric
    )
    
    # Plot threshold analysis for neural
    fig_neural, _, neural_optimal_thresholds = plot_threshold_analysis(
        true_labels, neural_scores_arr, 'Neural Network Scoring',
        save_path='neural_threshold_analysis.png'
    )
    
    # Calculate metrics with optimal threshold
    neural_pred = (neural_scores_arr >= best_neural_thresh).astype(int)
    
    results['neural'] = {
        'optimal_threshold': best_neural_thresh,
        'optimal_metric_value': best_neural_score,
        'all_optimal_thresholds': neural_optimal_thresholds,
        'accuracy': accuracy_score(true_labels, neural_pred),
        'precision': precision_score(true_labels, neural_pred, zero_division=0),
        'recall': recall_score(true_labels, neural_pred, zero_division=0),
        'f1': f1_score(true_labels, neural_pred, zero_division=0),
        'auc': roc_auc_score(true_labels, neural_scores_arr) if len(np.unique(true_labels)) > 1 else 0,
        'scores':neural_scores_arr,
        'predictions': neural_pred
    }
    
    # 4. COMBINED SCORING ACCURACY WITH OPTIMAL THRESHOLD
    print("\n3. Finding optimal threshold for combined scoring...")
    combined_scores_arr = np.array([
        next(details['combined_score'] for seg_key, _, details in ranked_segments if seg_key == key)
        for key in segment_keys
    ])
    
    # Find optimal threshold
    best_combined_thresh, best_combined_score, _, _ = find_optimal_threshold(
        true_labels, combined_scores_arr, metric=optimization_metric
    )
    
    # Plot threshold analysis for combined
    fig_combined, _, combined_optimal_thresholds = plot_threshold_analysis(
        true_labels, combined_scores_arr, 'Combined Scoring',
        save_path='combined_threshold_analysis.png'
    )
    
    # Calculate metrics with optimal threshold
    combined_pred = (combined_scores_arr >= best_combined_thresh).astype(int)
    
    results['combined'] = {
        'optimal_threshold': best_combined_thresh,
        'optimal_metric_value': best_combined_score,
        'all_optimal_thresholds': combined_optimal_thresholds,
        'accuracy': accuracy_score(true_labels, combined_pred),
        'precision': precision_score(true_labels, combined_pred, zero_division=0),
        'recall': recall_score(true_labels, combined_pred, zero_division=0),
        'f1': f1_score(true_labels, combined_pred, zero_division=0),
        'auc': roc_auc_score(true_labels, combined_scores_arr) if len(np.unique(true_labels)) > 1 else 0,
        'scores': combined_scores_arr,
        'predictions': combined_pred
    }
    
    # Store additional info
    results['true_labels'] = true_labels
    results['segment_keys'] = segment_keys
    results['optimization_metric'] = optimization_metric
    
    return results



def print_optimal_threshold_results(results):
    """Print formatted results including optimal thresholds."""
    if not results:
        print("No results to display!")
        return
    
    print("\n" + "="*80)
    print("OPTIMAL THRESHOLD EVALUATION RESULTS")
    print("="*80)
    print(f"Optimization metric: {results['optimization_metric']}")
    print(f"Dataset: {len(results['segment_keys'])} segments")
    
    methods = ['consensus', 'embedding', 'neural', 'combined']
    
    for method in methods:
        if method not in results:
            continue
            
        metrics = results[method]
        print(f"\n{method.upper()} SCORING:")
        print(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"  Optimized {results['optimization_metric']}: {metrics['optimal_metric_value']:.4f}")
        print(f"  Final Metrics:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1']:.4f}")
        print(f"    ROC AUC:   {metrics['auc']:.4f}")
        
        # Show all optimal thresholds for different metrics
        print(f"  All Optimal Thresholds:")
        for metric_name, thresh_info in metrics['all_optimal_thresholds'].items():
            print(f"    {metric_name}: {thresh_info['threshold']:.4f} (score: {thresh_info['score']:.4f})")



# Comprehensive plotting function
def generate_comprehensive_plots(consensus_scores, embedding_scores, neural_scores, ranked_segments, 
                                segment_sequences, accuracy_results=None, save_prefix="project4", union_keys=True):
    """Generate all plots for Project 4 analysis."""
    
    # Extract data for plotting
    segment_keys = [key for key, _, _ in ranked_segments]
    consensus_scores_aligned = [consensus_scores.get(key, {}).get('score', 0) for key in segment_keys]
    embedding_scores_aligned = [embedding_scores.get(key, 0) for key in segment_keys]
    neural_scores_aligned = [neural_scores.get(key, 0) for key in segment_keys]
    combined_scores_aligned = [score for _, score, _ in ranked_segments]
    sequence_lengths = [len(seq) for seq in segment_sequences.values()]


    # Normalize scores to [0,1] range for better comparison
    def normalize_scores(scores):
        min_val = min(scores)
        max_val = max(scores)
        if max_val == min_val:
            return [0.5] * len(scores)
        return [(x - min_val) / (max_val - min_val) for x in scores]

    consensus_scores = normalize_scores(consensus_scores_aligned)
    embedding_scores_list = normalize_scores(embedding_scores_aligned)
    neural_scores_list = normalize_scores(neural_scores_aligned)
    combined_scores = normalize_scores(combined_scores_aligned)

    # Create comprehensive plot grid
    if accuracy_results:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # Plot 1: Score distributions
    axes[0, 0].hist(consensus_scores, bins=30, alpha=0.7, color='blue', label='Consensus')
    axes[0, 0].hist(embedding_scores_list, bins=30, alpha=0.7, color='green', label='Embedding')
    axes[0, 0].hist(neural_scores_list, bins=30, alpha=0.7, color='purple', label='Neural')
    axes[0, 0].hist(combined_scores, bins=30, alpha=0.7, color='red', label='Combined')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Score Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Consensus vs Embedding 
    axes[0, 1].scatter(consensus_scores, embedding_scores_list, alpha=0.6, c=combined_scores, cmap='viridis')
    axes[0, 1].set_xlabel('Consensus Score')
    axes[0, 1].set_ylabel('Embedding Score')
    axes[0, 1].set_title('Consensus vs Embedding Scores')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sequence length distribution
    axes[0, 2].hist(sequence_lengths, bins=range(8, max(sequence_lengths)+2), alpha=0.7, color='purple')
    axes[0, 2].set_xlabel('Sequence Length')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Segment Length Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Top candidates
    top_20_scores = combined_scores[:20]
    axes[1, 0].bar(range(1, len(top_20_scores)+1), top_20_scores, color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Rank')
    axes[1, 0].set_ylabel('Combined Score')
    axes[1, 0].set_title('Top 20 Candidates')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Pattern frequency
    all_patterns = [match for _, _, details in ranked_segments for match in details['pattern_matches']]
    pattern_types = [match.split('@')[0] for match in all_patterns]
    pattern_counts = pd.Series(pattern_types).value_counts().head(10)
    
    axes[1, 1].barh(range(len(pattern_counts)), pattern_counts.values, color='cyan', alpha=0.7)
    axes[1, 1].set_yticks(range(len(pattern_counts)))
    axes[1, 1].set_yticklabels(pattern_counts.index)
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_title('Top 10 Pattern Matches')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Create a 1x3 grid of scatter plots within the same axes
    axes[1, 2].clear()
    gs = axes[1, 2].get_subplotspec().subgridspec(1, 3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax, scores, title, color in zip([ax1, ax2, ax3],
                                    [consensus_scores, embedding_scores_list, neural_scores_list],
                                    ['Consensus', 'Embedding', 'Neural'],
                                    ['red', 'blue', 'green']):
        ax.scatter(scores, combined_scores, alpha=0.6, color=color)
        ax.set_title(title)
        ax.set_xlabel('Score')
        ax.grid(True, alpha=0.3)
        
        # Add regression line
        z = np.polyfit(scores, combined_scores, 1)
        p = np.poly1d(z)
        ax.plot(scores, p(scores), color='black', linestyle='--')
        
    ax1.set_ylabel('Combined Score')
    axes[1, 2].axis('off')  # Hide original axes frame
    

    # Additional plots if accuracy results available
    if accuracy_results:
        # Plot 7: ROC curves
        methods = ['consensus', 'embedding', 'neural', 'combined']
        colors = ['blue', 'green', 'red', 'purple']
        
        axes[2, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        for method, color in zip(methods, colors):
            if method in accuracy_results:
                fpr, tpr, _ = roc_curve(accuracy_results['true_labels'], accuracy_results[method]['scores'])
                auc_score = accuracy_results[method]['auc']
                axes[2, 0].plot(fpr, tpr, color=color, lw=2, 
                               label=f'{method.title()} (AUC = {auc_score:.3f})')
        
        axes[2, 0].set_xlabel('False Positive Rate')
        axes[2, 0].set_ylabel('True Positive Rate')
        axes[2, 0].set_title('ROC Curves')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 8: Accuracy metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        method_names = list(accuracy_results.keys())[:4]  
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, method in enumerate(method_names):
            values = [accuracy_results[method][metric] for metric in metrics]
            axes[2, 1].bar(x + i*width, values, width, label=method.title(), alpha=0.7)
        
        axes[2, 1].set_xlabel('Metrics')
        axes[2, 1].set_ylabel('Score')
        axes[2, 1].set_title('Accuracy Metrics Comparison')
        axes[2, 1].set_xticks(x + width)
        axes[2, 1].set_xticklabels(metrics)
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()



# MAIN  PIPELINE FUNCTION
def run_comprehensive_project4_pipeline(
    data_dir="data",
    true_labels_file="zps_segments.jsonl",
    embedding_size=1280,
    embedding_layer=33,
    min_length=8,
    max_length=25,
    consensus_weight=0.4,
    embedding_weight=0.4,
    neural_net_weight=0.3,
    top_n_display=10,
    save_results=True,
    union_keys=True
):
    """
    Complete Project 4 pipeline with all analysis, visualization, and accuracy evaluation.
    """
    
    print("="*80)
    print("PROJECT 4: COMPREHENSIVE NES DETECTION PIPELINE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n STEP 1: Loading data...")
    
    # Load segmentation data
    with open(f"{data_dir}/protein_segments.pkl", "rb") as f:
        protein_segment_boundaries = pickle.load(f)
    
    with open(f"{data_dir}/protein_segment_embeddings.pkl", "rb") as f:
        protein_segment_embeddings = pickle.load(f)
    
    with open(f"{data_dir}/protein_sequences.pkl", "rb") as f:
        protein_sequences = pickle.load(f)
    
    # Load reference embeddings
    pos_pep, neg_pep, doubt_list = load_peptide_data_lists()
    model, alphabet, batch_converter, device = get_esm_model(embedding_size=embedding_size)
    
    positive_reference_embeddings = get_esm_embeddings(
        pos_pep, model, alphabet, batch_converter, device,
        embedding_layer=embedding_layer, sequence_embedding=True
    )
    
    negative_reference_embeddings = get_esm_embeddings(
        neg_pep, model, alphabet, batch_converter, device,
        embedding_layer=embedding_layer, sequence_embedding=True
    )
    
    # Load true labels if available
    true_labels_dict = load_true_labels(true_labels_file)
    
    print(f" Loaded {len(protein_segment_boundaries)} proteins with segment boundaries")
    print(f" Loaded {len(protein_segment_embeddings)} segment embeddings")
    print(f" Loaded {len(protein_sequences)} protein sequences")
    print(f" Generated {len(positive_reference_embeddings)} positive reference embeddings")
    print(f" Generated {len(negative_reference_embeddings)} negative reference embeddings")
    
    # ========================================================================
    # STEP 2: FILTER AND EXTRACT SEGMENTS
    # ========================================================================
    print(f"\n STEP 2: Filtering segments by length ({min_length}-{max_length} amino acids)...")
    
    filtered_boundaries = filter_segments_by_length(
        protein_segment_boundaries, min_length=min_length, max_length=max_length
    )
    
    segment_sequences = extract_segment_sequences(protein_sequences, filtered_boundaries)
    
    total_segments_before = sum(len(segments) for segments in protein_segment_boundaries.values() if segments != "Failed")
    total_segments_after = len(segment_sequences)
    
    print(f" Segments before filtering: {total_segments_before}")
    print(f" Segments after filtering: {total_segments_after}")
    print(f" Filtering efficiency: {total_segments_after/total_segments_before*100:.1f}%")
    
    # ========================================================================
    # STEP 3: CONSENSUS PATTERN SCORING
    # ========================================================================
    print("\n STEP 3: Scoring segments with NES consensus patterns...")
    
    consensus_results = score_nes_consensus_pattern(segment_sequences)
    consensus_scores = {key: result['score'] for key, result in consensus_results.items()}
    
    print(f" Scored {len(consensus_results)} segments")
    print(f" Score range: {min(consensus_scores.values()):.1f} - {max(consensus_scores.values()):.1f}")
    print(f" Average score: {np.mean(list(consensus_scores.values())):.2f}")
    
    # Show top 5 consensus matches
    top_consensus = sorted(consensus_results.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
    print(f"\n Top 5 consensus pattern matches:")
    for i, (segment_key, result) in enumerate(top_consensus):
        sequence = segment_sequences[segment_key]
        print(f"  {i+1}. {segment_key}")
        print(f"     Sequence: {sequence}")
        print(f"     Score: {result['score']}")
        print(f"     Patterns: {', '.join(result['matches'][:5])}{'...' if len(result['matches']) > 5 else ''}")
    
    # ========================================================================
    # STEP 4: EMBEDDING SIMILARITY SCORING
    # ========================================================================
    print("\n STEP 4: Scoring segments with embedding similarity...")
    
    filtered_embeddings = {
        key: emb for key, emb in protein_segment_embeddings.items() 
        if key in segment_sequences
    }
    
    embedding_scores = score_segments_with_embeddings(
        filtered_embeddings, positive_reference_embeddings, negative_reference_embeddings
    )
    
    print(f" Scored {len(embedding_scores)} segments with embedding similarity")
    print(f" Score range: {min(embedding_scores.values()):.3f} - {max(embedding_scores.values()):.3f}")
    print(f" Average score: {np.mean(list(embedding_scores.values())):.3f}")
    
    # Show top 5 embedding matches
    top_embedding = sorted(embedding_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n Top 5 embedding similarity matches:")
    for i, (segment_key, score) in enumerate(top_embedding):
        sequence = segment_sequences[segment_key]
        print(f"  {i+1}. {segment_key}")
        print(f"     Sequence: {sequence}")
        print(f"     Embedding score: {score:.4f}")
    
    # ========================================================================
    # STEP 5: NEURAL NETWORK CLASSIFICATION SCORING
    # ========================================================================
    print("\n STEP 5: Scoring segments with neural network classification...")
    neural_net_scores = None
    if union_keys:
        neural_net_scores = pickle.load(open(f"data/network_all_scores.pkl", "rb"))
        neural_net_scores = {
            key: score for key, score in neural_net_scores.items()
            if key in segment_sequences
        }
    else:
        neural_net_scores = pickle.load(open(f"data/network_test_scores.pkl", "rb"))
        neural_net_scores = {
            key: score for key, score in neural_net_scores.items()
            if key in segment_sequences and key in consensus_results
        }
    

    print(f" Scored {len(neural_net_scores)} segments")
    print(f" Score range: {min(neural_net_scores.values()):.3f} - {max(neural_net_scores.values()):.3f}")
    print(f" Average score: {np.mean(list(neural_net_scores.values())):.3f}")
    
    # Show top 5 
    top_neural = sorted(neural_net_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n Top 5 neural net:")
    for i, (segment_key, score) in enumerate(top_neural):
        sequence = segment_sequences[segment_key]
        print(f"  {i+1}. {segment_key}")
        print(f"     Sequence: {sequence}")
        print(f"     Neural Network Score: {score:.4f}")

    # ========================================================================
    # STEP 6: COMBINE SCORES AND RANK
    # ========================================================================
    print(f"\n STEP 6: Combining scores and ranking segments...")
    ranked_segments = combine_scores_and_rank(
        consensus_results, embedding_scores, neural_net_scores,
        consensus_weight=consensus_weight, embedding_weight=embedding_weight, neural_net_weight=neural_net_weight, union_keys=True
    )
    
    print(f" Ranked {len(ranked_segments)} segments")
    print(f" Weighting: {consensus_weight:.1f} consensus + {embedding_weight:.1f} embedding + {neural_net_weight:.1f} neural")
    
    # ========================================================================
    # STEP 7: DISPLAY TOP CANDIDATES
    # ========================================================================
    print(f"\n STEP 7: TOP {top_n_display} NES CANDIDATES")
    print("="*80)
    
    for i, (segment_key, combined_score, details) in enumerate(ranked_segments[:top_n_display]):
        sequence = segment_sequences[segment_key]
        print(f"\nRANK #{i+1}: {segment_key}")
        print(f"Sequence: {sequence}")
        print(f"Length: {len(sequence)} amino acids")
        print(f"Combined Score: {combined_score:.4f}")
        print(f"  • Consensus Score: {details['raw_consensus']} (patterns: {len(details['pattern_matches'])})")
        print(f"  • Embedding Score: {details['raw_embedding']:.4f}")
        print(f"  • Neural Network Score: {details['raw_neural_net']:.4f}")
        print(f"  • Pattern Matches: {', '.join(details['pattern_matches'][:3])}{'...' if len(details['pattern_matches']) > 3 else ''}")
        print("-" * 60)
    
    # ========================================================================
    # STEP 8: ACCURACY EVALUATION
    # ========================================================================
    accuracy_results = None
    if true_labels_dict:
        print(f"\n STEP 8: Accuracy Evaluation")
        print("="*50)
        

        accuracy_results = evaluate_accuracy(
            segment_sequences, consensus_results, embedding_scores, neural_net_scores, 
            ranked_segments, true_labels_dict, optimization_metric='f1', union_keys=True
        )
        
        print_optimal_threshold_results(accuracy_results)

        if accuracy_results:
            for method in ['consensus', 'embedding', 'neural', 'combined']:
                if method in accuracy_results:
                    metrics = accuracy_results[method]
                    print(f"\n{method.upper()} SCORING:")
                    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall:    {metrics['recall']:.4f}")
                    print(f"  F1-Score:  {metrics['f1']:.4f}")
                    print(f"  ROC AUC:   {metrics['auc']:.4f}")
    else:
        print(f"\n STEP 8: Accuracy evaluation skipped (no true labels available)")
    
    # ========================================================================
    # STEP 9: GENERATE ROC CURVES
    # ========================================================================
    if accuracy_results:
        print(f"\n STEP 9: Generating individual ROC curves")
        
        # Generate individual ROC curves using your existing function
        for method in ['consensus', 'embedding', 'neural', 'combined']:
            if method in accuracy_results:
                plot_roc_curve(
                    accuracy_results['true_labels'], 
                    accuracy_results[method]['scores'],
                    out_file_path=f"project4_{method}_roc_curve.png"
                )
                print(f" {method.title()} ROC curve saved")
    
    # ========================================================================
    # STEP 10: COMPREHENSIVE VISUALIZATION
    # ========================================================================
    print(f"\n STEP 10: Generating comprehensive analysis plots...")
    
    generate_comprehensive_plots(
        consensus_results, embedding_scores, neural_net_scores, ranked_segments,
        segment_sequences, accuracy_results, save_prefix="project4"
    )
    
    print(" Comprehensive analysis plots saved")
    
    # ========================================================================
    # STEP 11: SUMMARY STATISTICS AND SAVE RESULTS
    # ========================================================================
    print(f"\n STEP 11: Summary Statistics")
    print("="*50)
    
    # Score distributions
    all_combined_scores = [score for _, score, _ in ranked_segments]
    all_consensus_scores = [details['raw_consensus'] for _, _, details in ranked_segments]
    all_embedding_scores = [details['raw_embedding'] for _, _, details in ranked_segments]
    all_neural_scores = [details['raw_neural_net'] for _, _, details in ranked_segments]
    
    print(f"Combined Scores:")
    print(f"  Range: {min(all_combined_scores):.4f} - {max(all_combined_scores):.4f}")
    print(f"  Mean: {np.mean(all_combined_scores):.4f}")
    print(f"  Std: {np.std(all_combined_scores):.4f}")
    
    print(f"\nConsensus Scores:")
    print(f"  Range: {min(all_consensus_scores)} - {max(all_consensus_scores)}")
    print(f"  Mean: {np.mean(all_consensus_scores):.2f}")
    
    print(f"\nEmbedding Scores:")
    print(f"  Range: {min(all_embedding_scores):.4f} - {max(all_embedding_scores):.4f}")
    print(f"  Mean: {np.mean(all_embedding_scores):.4f}")

    print(f"\nNeural Network Scores:")
    print(f"  Range: {min(all_neural_scores):.4f} - {max(all_neural_scores):.4f}")
    print(f"  Mean: {np.mean(all_neural_scores):.4f}")
    
    # Pattern analysis
    all_pattern_matches = [match for _, _, details in ranked_segments for match in details['pattern_matches']]
    pattern_counts = {}
    for match in all_pattern_matches:
        pattern_type = match.split('@')[0]
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    print(f"\nTop Pattern Matches:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pattern}: {count} matches")
    
    # Save results if requested
    if save_results:
        print(f"\n Saving results...")
        
        # Prepare results for CSV
        results_data = []
        for i, (segment_key, combined_score, details) in enumerate(ranked_segments):
            protein_id = segment_key.split()[0]
            positions = segment_key.split()[1]
            sequence = segment_sequences[segment_key]
            
            results_data.append({
                'rank': i + 1,
                'segment_id': segment_key,
                'protein_id': protein_id,
                'positions': positions,
                'sequence': sequence,
                'sequence_length': len(sequence),
                'combined_score': combined_score,
                'consensus_score': details['raw_consensus'],
                'embedding_score': details['raw_embedding'],
                'neural_scor': details['raw_neural_net'],
                'num_pattern_matches': len(details['pattern_matches']),
                'pattern_matches': '; '.join(details['pattern_matches'])
            })
        
        # Save to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('project4_nes_candidates.csv', index=False)
        print(f" Results saved to 'project4_nes_candidates.csv'")
        
        # Save accuracy results if available
        if accuracy_results:
            accuracy_summary = {}
            for method in ['consensus', 'embedding', 'neural', 'combined']:
                if method in accuracy_results:
                    accuracy_summary[method] = {k: v for k, v in accuracy_results[method].items() 
                                              if k not in ['scores', 'predictions']}
            
            accuracy_df = pd.DataFrame(accuracy_summary).T
            accuracy_df.to_csv('project4_accuracy_results.csv')
            print(f" Accuracy results saved to 'project4_accuracy_results.csv'")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" PROJECT 4 COMPREHENSIVE PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f" Top NES candidate: {ranked_segments[0][0]}")
    print(f" Sequence: {segment_sequences[ranked_segments[0][0]]}")
    print(f" Combined score: {ranked_segments[0][1]:.4f}")
    print(f" Results saved to: project4_nes_candidates.csv")
    print(f" Plots saved with prefix: project4_")
    
    if accuracy_results:
        best_method = max(['consensus', 'embedding', 'neural', 'combined'], 
                         key=lambda m: accuracy_results[m]['auc'] if m in accuracy_results else 0)
        best_auc = accuracy_results[best_method]['auc']
        print(f" Best performing method: {best_method} (AUC = {best_auc:.4f})")
    
    return {
        'ranked_segments': ranked_segments,
        'segment_sequences': segment_sequences,
        'consensus_results': consensus_results,
        'embedding_scores': embedding_scores,
        'neural_scores': neural_net_scores,
        'accuracy_results': accuracy_results,
        'results_dataframe': results_df if save_results else None
    }



# Run the comprehensive pipeline
if __name__ == "__main__":
    results = run_comprehensive_project4_pipeline(
        data_dir="data",
        true_labels_file="data/zps_segments.jsonl",
        embedding_size=1280,
        embedding_layer=33,
        min_length=8,
        max_length=25,
        consensus_weight=0.6,
        embedding_weight=0.4,
        neural_net_weight=0.3,
        top_n_display=10,
        save_results=True,
        union_keys=False
    )
