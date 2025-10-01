"""
Tox21 Dataset Structure Analysis
Systematic exploration of the Tox21 dataset structure before visualization
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# PyTorch Geometric imports
from torch_geometric.datasets import MoleculeNet

def comprehensive_dataset_analysis():
    """Comprehensive analysis of Tox21 dataset structure"""
    
    print("="*80)
    print("TOX21 DATASET STRUCTURE ANALYSIS")
    print("="*80)
    
    # Load dataset
    print("1. LOADING DATASET...")
    print("-" * 40)
    dataset = MoleculeNet(root='./data', name='Tox21')
    
    print(f"✓ Dataset loaded successfully")
    print(f"✓ Total samples: {len(dataset)}")
    print(f"✓ Number of tasks: {dataset.num_tasks}")
    print(f"✓ Number of node features: {dataset.num_node_features}")
    print(f"✓ Number of edge features: {dataset.num_edge_features}")
    
    # Analyze first few samples
    print("\n2. SAMPLE STRUCTURE ANALYSIS...")
    print("-" * 40)
    
    sample_info = []
    for i in range(min(10, len(dataset))):
        data = dataset[i]
        info = {
            'sample_idx': i,
            'num_nodes': data.x.shape[0],
            'num_edges': data.edge_index.shape[1],
            'node_features_shape': data.x.shape,
            'edge_index_shape': data.edge_index.shape,
            'edge_attr_shape': data.edge_attr.shape if data.edge_attr is not None else None,
            'labels_shape': data.y.shape,
            'labels_values': data.y.numpy(),
            'has_smiles': hasattr(data, 'smiles'),
            'smiles': data.smiles if hasattr(data, 'smiles') else None
        }
        sample_info.append(info)
        
        if i < 3:  # Print detailed info for first 3 samples
            print(f"\nSample {i}:")
            print(f"  - Nodes (atoms): {info['num_nodes']}")
            print(f"  - Edges (bonds): {info['num_edges']}")
            print(f"  - Node features shape: {info['node_features_shape']}")
            print(f"  - Edge index shape: {info['edge_index_shape']}")
            print(f"  - Edge attr shape: {info['edge_attr_shape']}")
            print(f"  - Labels shape: {info['labels_shape']}")
            print(f"  - Labels: {info['labels_values']}")
            print(f"  - Has SMILES: {info['has_smiles']}")
            if info['has_smiles']:
                print(f"  - SMILES: {info['smiles']}")
    
    # Analyze label structure across all samples
    print("\n3. LABEL STRUCTURE ANALYSIS...")
    print("-" * 40)
    
    all_labels = []
    label_shapes = []
    missing_patterns = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        labels = data.y.numpy()
        all_labels.append(labels)
        label_shapes.append(labels.shape)
        
        # Analyze missing data pattern
        if len(labels.shape) > 0 and labels.shape[0] > 1:
            missing_mask = np.isnan(labels)
            missing_patterns.append(missing_mask)
    
    all_labels = np.array(all_labels)
    print(f"✓ All labels shape: {all_labels.shape}")
    print(f"✓ Unique label shapes: {list(set(label_shapes))}")
    
    # Check if this is the expected 12-task format
    if all_labels.shape[1] == 12:
        print("✓ Confirmed: 12-task Tox21 dataset")
        task_names = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
            'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
    else:
        print(f"⚠ Unexpected number of tasks: {all_labels.shape[1]}")
        task_names = [f'Task_{i}' for i in range(all_labels.shape[1])]
    
    # Analyze each task
    print("\n4. TASK-SPECIFIC ANALYSIS...")
    print("-" * 40)
    
    task_stats = []
    for task_idx, task_name in enumerate(task_names):
        task_labels = all_labels[:, task_idx]
        
        # Count values
        total_samples = len(task_labels)
        missing_count = np.sum(np.isnan(task_labels))
        valid_count = total_samples - missing_count
        
        if valid_count > 0:
            valid_labels = task_labels[~np.isnan(task_labels)]
            inactive_count = np.sum(valid_labels == 0)
            active_count = np.sum(valid_labels == 1)
            other_values = valid_count - inactive_count - active_count
            
            stats = {
                'Task': task_name,
                'Total': total_samples,
                'Valid': valid_count,
                'Missing': missing_count,
                'Inactive (0)': inactive_count,
                'Active (1)': active_count,
                'Other values': other_values,
                'Missing %': (missing_count / total_samples) * 100,
                'Active %': (active_count / valid_count) * 100 if valid_count > 0 else 0
            }
            task_stats.append(stats)
            
            print(f"{task_name:15} | Valid: {valid_count:4d} | Missing: {missing_count:4d} | Active: {active_count:3d} ({active_count/valid_count*100:5.1f}%) | Inactive: {inactive_count:4d}")
    
    # Create summary DataFrame
    stats_df = pd.DataFrame(task_stats)
    
    # Analyze node features
    print("\n5. NODE FEATURE ANALYSIS...")
    print("-" * 40)
    
    sample_node_features = dataset[0].x
    print(f"✓ Node feature dimensionality: {sample_node_features.shape[1]}")
    print(f"✓ Node feature data type: {sample_node_features.dtype}")
    print(f"✓ Sample node features (first atom):")
    print(f"  {sample_node_features[0].numpy()}")
    
    # Check if features are one-hot encoded
    unique_values = torch.unique(sample_node_features)
    print(f"✓ Unique values in node features: {unique_values.numpy()}")
    
    if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
        print("✓ Node features appear to be one-hot encoded")
    
    # Analyze edge features
    print("\n6. EDGE FEATURE ANALYSIS...")
    print("-" * 40)
    
    sample_edge_attr = dataset[0].edge_attr
    if sample_edge_attr is not None:
        print(f"✓ Edge feature dimensionality: {sample_edge_attr.shape[1]}")
        print(f"✓ Edge feature data type: {sample_edge_attr.dtype}")
        print(f"✓ Sample edge features (first bond):")
        print(f"  {sample_edge_attr[0].numpy()}")
        
        unique_edge_values = torch.unique(sample_edge_attr)
        print(f"✓ Unique values in edge features: {unique_edge_values.numpy()}")
    else:
        print("⚠ No edge features found")
    
    # Analyze molecular sizes
    print("\n7. MOLECULAR SIZE ANALYSIS...")
    print("-" * 40)
    
    node_counts = []
    edge_counts = []
    
    for i in range(min(1000, len(dataset))):  # Sample for speed
        data = dataset[i]
        node_counts.append(data.x.shape[0])
        edge_counts.append(data.edge_index.shape[1] // 2)  # Undirected edges
    
    print(f"✓ Analyzed {len(node_counts)} molecules")
    print(f"✓ Atoms per molecule: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f} (min: {np.min(node_counts)}, max: {np.max(node_counts)})")
    print(f"✓ Bonds per molecule: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f} (min: {np.min(edge_counts)}, max: {np.max(edge_counts)})")
    
    # Check for SMILES availability
    print("\n8. SMILES AVAILABILITY...")
    print("-" * 40)
    
    smiles_count = 0
    sample_smiles = []
    
    for i in range(min(100, len(dataset))):
        data = dataset[i]
        if hasattr(data, 'smiles'):
            smiles_count += 1
            if len(sample_smiles) < 5:
                sample_smiles.append(data.smiles)
    
    print(f"✓ SMILES available: {smiles_count}/{min(100, len(dataset))} samples checked")
    if sample_smiles:
        print("✓ Sample SMILES:")
        for i, smiles in enumerate(sample_smiles):
            print(f"  {i+1}. {smiles}")
    
    # Summary
    print("\n9. SUMMARY...")
    print("-" * 40)
    print(f"✓ Dataset successfully analyzed")
    print(f"✓ Structure: {len(dataset)} molecules × {dataset.num_tasks} tasks")
    print(f"✓ Node features: {dataset.num_node_features}D")
    print(f"✓ Edge features: {dataset.num_edge_features}D")
    print(f"✓ Average molecule size: {np.mean(node_counts):.1f} atoms")
    print(f"✓ SMILES available: {'Yes' if smiles_count > 0 else 'No'}")
    print(f"✓ Missing data: Significant (varies by task)")
    
    return dataset, stats_df, task_names, all_labels

def analyze_missing_data_patterns(all_labels, task_names):
    """Analyze patterns in missing data"""
    print("\n10. MISSING DATA PATTERN ANALYSIS...")
    print("-" * 40)
    
    # Calculate missing data correlation
    missing_matrix = np.isnan(all_labels)
    
    # Count missing per sample
    missing_per_sample = np.sum(missing_matrix, axis=1)
    print(f"✓ Samples with no missing labels: {np.sum(missing_per_sample == 0)}")
    print(f"✓ Samples with all missing labels: {np.sum(missing_per_sample == len(task_names))}")
    print(f"✓ Average missing labels per sample: {np.mean(missing_per_sample):.1f}")
    
    # Missing data heatmap
    plt.figure(figsize=(12, 8))
    sample_indices = np.random.choice(len(all_labels), min(500, len(all_labels)), replace=False)
    sample_indices.sort()
    
    plt.imshow(missing_matrix[sample_indices].T, cmap='RdYlBu', aspect='auto')
    plt.xlabel('Sample Index')
    plt.ylabel('Task')
    plt.title('Missing Data Pattern (Red = Missing, Blue = Present)')
    plt.yticks(range(len(task_names)), task_names)
    plt.colorbar(label='Missing (1) / Present (0)')
    plt.tight_layout()
    plt.show()
    
    return missing_matrix

def main():
    """Main analysis function"""
    # Run comprehensive analysis
    dataset, stats_df, task_names, all_labels = comprehensive_dataset_analysis()
    
    # Analyze missing data patterns
    missing_matrix = analyze_missing_data_patterns(all_labels, task_names)
    
    print("\n" + "="*80)
    print("STRUCTURE ANALYSIS COMPLETE!")
    print("="*80)
    print("Ready for visualization phase...")
    
    return {
        'dataset': dataset,
        'stats_df': stats_df,
        'task_names': task_names,
        'all_labels': all_labels,
        'missing_matrix': missing_matrix
    }

if __name__ == "__main__":
    results = main()