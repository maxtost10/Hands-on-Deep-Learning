#%%
"""
Tox21 Dataset Visualization Script
Based on the discovered structure: 7823 molecules, 12 tasks, labels shape (1, 12)
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
from torch_geometric.utils import to_networkx

# RDKit imports for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available. Install with: conda install -c rdkit rdkit")
    RDKIT_AVAILABLE = False

# NetworkX for graph visualization
import networkx as nx

# Set style
plt.style.use('default')
sns.set_palette("husl")

#%%
def load_and_prepare_data():
    """Load dataset and prepare labels for analysis"""
    print("Loading Tox21 dataset...")
    dataset = MoleculeNet(root='./data', name='Tox21')
    
    # Extract all labels - reshape from (n, 1, 12) to (n, 12)
    all_labels = []
    for i in range(len(dataset)):
        labels = dataset[i].y.numpy().reshape(-1)  # Flatten (1, 12) to (12,)
        all_labels.append(labels)
    
    all_labels = np.array(all_labels)  # Shape: (7823, 12)
    
    # Define task names
    task_names = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
        'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    
    print(f"✓ Dataset loaded: {len(dataset)} molecules")
    print(f"✓ Labels reshaped: {all_labels.shape}")
    print(f"✓ Task names: {task_names}")
    
    return dataset, all_labels, task_names

# Load data
dataset, all_labels, task_names = load_and_prepare_data()

#%%
def analyze_class_distributions(all_labels, task_names, save_figures=True):
    """Analyze and visualize class distributions for all tasks"""
    import os
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Create figures directory if it doesn't exist
    if save_figures:
        os.makedirs('./Figures', exist_ok=True)
    
    # Calculate statistics for each task
    task_stats = []
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Tox21 Dataset: Class Distribution Across All 12 Tasks', fontsize=16, y=0.98)
    axes = axes.flatten()
    
    for task_idx, task_name in enumerate(task_names):
        task_labels = all_labels[:, task_idx]
        
        # Count classes
        total_samples = len(task_labels)
        missing_mask = np.isnan(task_labels)
        missing_count = np.sum(missing_mask)
        valid_labels = task_labels[~missing_mask]
        valid_count = len(valid_labels)
        
        if valid_count > 0:
            inactive_count = np.sum(valid_labels == 0)
            active_count = np.sum(valid_labels == 1)
            other_count = valid_count - inactive_count - active_count
            
            # Store statistics
            stats = {
                'Task': task_name,
                'Total': total_samples,
                'Valid': valid_count,
                'Missing': missing_count,
                'Inactive (0)': inactive_count,
                'Active (1)': active_count,
                'Other': other_count,
                'Missing %': (missing_count / total_samples) * 100,
                'Active %': (active_count / valid_count) * 100 if valid_count > 0 else 0
            }
            task_stats.append(stats)
            
            # Create bar plot
            ax = axes[task_idx]
            counts = [inactive_count, active_count, missing_count]
            labels = ['Non-toxic', 'Toxic', 'Missing']
            colors = ['lightblue', 'salmon', 'lightgray']
            
            bars = ax.bar(labels, counts, color=colors, alpha=0.8)
            ax.set_title(f'{task_name}\n(n={valid_count} valid)', fontsize=11)
            ax.set_ylabel('Count', fontsize=10)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(counts)*0.01,
                           f'{count}', ha='center', va='bottom', fontsize=9)
            
            # Add active percentage
            if active_count > 0:
                ax.text(0.98, 0.98, f'{active_count/valid_count*100:.1f}% active', 
                       transform=ax.transAxes, ha='right', va='top', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=8)
            
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_figures:
        filename = f'class_distributions.png'
        filepath = os.path.join('./Figures', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved: {filepath}")
    
    plt.show()
    
    # Print summary table
    stats_df = pd.DataFrame(task_stats)
    print("\nTask Summary Statistics:")
    print("=" * 100)
    print(stats_df.to_string(index=False))
    
    return stats_df

# Analyze class distributions
stats_df = analyze_class_distributions(all_labels, task_names)

#%%
def visualize_molecular_examples(dataset, all_labels, task_names, num_examples=6, save_figures=True):
    """Visualize molecular graph examples for toxic vs non-toxic"""
    import os
    
    print("\n" + "="*60)
    print("MOLECULAR GRAPH VISUALIZATION")
    print("="*60)
    
    # Create figures directory if it doesn't exist
    if save_figures:
        os.makedirs('./Figures', exist_ok=True)
    
    # Focus on NR-AhR task (liver toxicity, index 2)
    focus_task_idx = 2
    focus_task_name = task_names[focus_task_idx]
    
    print(f"Focusing on task: {focus_task_name} (liver toxicity)")
    
    # Find examples
    task_labels = all_labels[:, focus_task_idx]
    valid_mask = ~np.isnan(task_labels)
    
    toxic_indices = np.where((task_labels == 1) & valid_mask)[0]
    non_toxic_indices = np.where((task_labels == 0) & valid_mask)[0]
    
    print(f"Found {len(toxic_indices)} toxic and {len(non_toxic_indices)} non-toxic examples")
    
    # Sample examples
    np.random.seed(42)
    sample_toxic = np.random.choice(toxic_indices, min(num_examples//2, len(toxic_indices)), replace=False)
    sample_non_toxic = np.random.choice(non_toxic_indices, min(num_examples//2, len(non_toxic_indices)), replace=False)
    
    # Print one raw datapoint example
    print("\n" + "-"*40)
    print("RAW DATAPOINT EXAMPLE:")
    print("-"*40)
    example_idx = sample_toxic[0]
    example_data = dataset[example_idx]
    print(f"Molecule index: {example_idx}")
    print(f"SMILES: {example_data.smiles}")
    print(f"Node features shape: {example_data.x.shape}")
    print(f"Edge index shape: {example_data.edge_index.shape}")
    print(f"Edge attributes shape: {example_data.edge_attr.shape}")
    print(f"Labels shape: {example_data.y.shape}")
    print(f"All labels: {example_data.y.numpy().flatten()}")
    print(f"Label for {focus_task_name}: {example_data.y.numpy().flatten()[focus_task_idx]}")
    print(f"First 3 node features:")
    for i in range(min(3, example_data.x.shape[0])):
        print(f"  Node {i}: {example_data.x[i].numpy()}")
    print(f"First 3 edge indices:")
    for i in range(min(3, example_data.edge_index.shape[1])):
        print(f"  Edge {i}: {example_data.edge_index[:, i].numpy()} (connects atoms {example_data.edge_index[0, i].item()} → {example_data.edge_index[1, i].item()})")
    print(f"First 3 edge attributes:")
    for i in range(min(3, example_data.edge_attr.shape[0])):
        print(f"  Edge attr {i}: {example_data.edge_attr[i].numpy()}")
    print("-"*40)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_examples//2, figsize=(15, 8))
    fig.suptitle(f'Molecular Graphs - {focus_task_name} Task (Liver Toxicity)', fontsize=16)
    
    # Plot toxic examples
    for i, mol_idx in enumerate(sample_toxic):
        data = dataset[mol_idx]
        G = to_networkx(data, to_undirected=True)
        
        ax = axes[0, i]
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        nx.draw(G, pos, ax=ax, node_color='salmon', node_size=120, 
               with_labels=False, edge_color='gray', alpha=0.8, width=1.5)
        ax.set_title(f'TOXIC #{mol_idx}\n{G.number_of_nodes()} atoms, {G.number_of_edges()} bonds\n{data.smiles[:30]}...', fontsize=10)
        ax.axis('off')
    
    # Plot non-toxic examples
    for i, mol_idx in enumerate(sample_non_toxic):
        data = dataset[mol_idx]
        G = to_networkx(data, to_undirected=True)
        
        ax = axes[1, i]
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        nx.draw(G, pos, ax=ax, node_color='lightblue', node_size=120,
               with_labels=False, edge_color='gray', alpha=0.8, width=1.5)
        ax.set_title(f'NON-TOXIC #{mol_idx}\n{G.number_of_nodes()} atoms, {G.number_of_edges()} bonds\n{data.smiles[:30]}...', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_figures:
        filename = f'molecular_graphs_{focus_task_name}.png'
        filepath = os.path.join('./Figures', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved: {filepath}")
    
    plt.show()
    
    return sample_toxic, sample_non_toxic

# Visualize molecular examples
toxic_examples, non_toxic_examples = visualize_molecular_examples(dataset, all_labels, task_names)

#%%
def analyze_molecular_properties(dataset, all_labels, task_names):
    """Analyze molecular properties and their relationship to toxicity"""
    print("\n" + "="*60)
    print("MOLECULAR PROPERTY ANALYSIS")
    print("="*60)
    
    # Focus on NR-AhR task
    focus_task_idx = 2
    focus_task_name = task_names[focus_task_idx]
    
    # Collect molecular properties
    properties = []
    toxicity_labels = []
    
    for i in range(min(2000, len(dataset))):  # Sample for speed
        data = dataset[i]
        label = all_labels[i, focus_task_idx]
        
        if not np.isnan(label):  # Only include valid labels
            # Graph-based properties
            num_atoms = data.x.shape[0]
            num_bonds = data.edge_index.shape[1] // 2
            
            props = {
                'num_atoms': num_atoms,
                'num_bonds': num_bonds,
                'bond_atom_ratio': num_bonds / num_atoms if num_atoms > 0 else 0
            }
            
            # RDKit properties if available
            if RDKIT_AVAILABLE and hasattr(data, 'smiles'):
                try:
                    mol = Chem.MolFromSmiles(data.smiles)
                    if mol:
                        props.update({
                            'mol_weight': Descriptors.MolWt(mol),
                            'logp': Descriptors.MolLogP(mol),
                            'hbd': Descriptors.NumHDonors(mol),
                            'hba': Descriptors.NumHAcceptors(mol),
                            'tpsa': Descriptors.TPSA(mol),
                            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                            'aromatic_rings': Descriptors.NumAromaticRings(mol)
                        })
                except:
                    continue
            
            properties.append(props)
            toxicity_labels.append(int(label))
    
    if not properties:
        print("No valid molecular properties found")
        return
    
    # Create DataFrame
    props_df = pd.DataFrame(properties)
    props_df['toxicity'] = toxicity_labels
    
    print(f"Analyzed {len(props_df)} molecules with valid {focus_task_name} labels")
    
    # Create comparison plots
    if RDKIT_AVAILABLE and 'mol_weight' in props_df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Molecular Properties vs {focus_task_name} Toxicity', fontsize=16)
        axes = axes.flatten()
        
        properties_to_plot = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa', 'num_atoms']
        
        for i, prop in enumerate(properties_to_plot):
            if prop in props_df.columns:
                ax = axes[i]
                
                # Box plot
                toxic_data = props_df[props_df['toxicity'] == 1][prop]
                non_toxic_data = props_df[props_df['toxicity'] == 0][prop]
                
                box_data = [non_toxic_data, toxic_data]
                box = ax.boxplot(box_data, labels=['Non-toxic', 'Toxic'], patch_artist=True)
                box['boxes'][0].set_facecolor('lightblue')
                box['boxes'][1].set_facecolor('salmon')
                
                ax.set_title(f'{prop.replace("_", " ").title()}')
                ax.set_ylabel('Value')
                
                # Add statistics
                if len(toxic_data) > 0 and len(non_toxic_data) > 0:
                    from scipy.stats import mannwhitneyu
                    try:
                        stat, p_value = mannwhitneyu(toxic_data, non_toxic_data)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        ax.text(0.98, 0.98, f'p={p_value:.3f} {significance}', 
                               transform=ax.transAxes, ha='right', va='top',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    except:
                        pass
    else:
        # Simple graph-based properties only
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'Graph Properties vs {focus_task_name} Toxicity', fontsize=16)
        
        graph_props = ['num_atoms', 'num_bonds', 'bond_atom_ratio']
        
        for i, prop in enumerate(graph_props):
            ax = axes[i]
            
            toxic_data = props_df[props_df['toxicity'] == 1][prop]
            non_toxic_data = props_df[props_df['toxicity'] == 0][prop]
            
            box_data = [non_toxic_data, toxic_data]
            box = ax.boxplot(box_data, labels=['Non-toxic', 'Toxic'], patch_artist=True)
            box['boxes'][0].set_facecolor('lightblue')
            box['boxes'][1].set_facecolor('salmon')
            
            ax.set_title(f'{prop.replace("_", " ").title()}')
            ax.set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nProperty comparison for {focus_task_name}:")
    print("=" * 50)
    toxic_summary = props_df[props_df['toxicity'] == 1].describe()
    non_toxic_summary = props_df[props_df['toxicity'] == 0].describe()
    
    print("TOXIC molecules:")
    print(toxic_summary)
    print("\nNON-TOXIC molecules:")
    print(non_toxic_summary)

# Analyze molecular properties
analyze_molecular_properties(dataset, all_labels, task_names)

#%%
def analyze_task_correlations(all_labels, task_names, save_figures=True):
    """Analyze correlations between different toxicity tasks"""
    import os
    from datetime import datetime
    
    print("\n" + "="*60)
    print("TASK CORRELATION ANALYSIS")
    print("="*60)
    
    # Create figures directory if it doesn't exist
    if save_figures:
        os.makedirs('./Figures', exist_ok=True)
    
    # Calculate correlation matrix for valid labels only
    n_tasks = len(task_names)
    corr_matrix = np.full((n_tasks, n_tasks), np.nan)
    
    for i in range(n_tasks):
        for j in range(n_tasks):
            task_i = all_labels[:, i]
            task_j = all_labels[:, j]
            
            # Find samples with valid labels for both tasks
            valid_mask = ~(np.isnan(task_i) | np.isnan(task_j))
            
            if np.sum(valid_mask) > 10:  # Need at least 10 samples
                valid_i = task_i[valid_mask]
                valid_j = task_j[valid_mask]
                
                if len(np.unique(valid_i)) > 1 and len(np.unique(valid_j)) > 1:
                    corr = np.corrcoef(valid_i, valid_j)[0, 1]
                    corr_matrix[i, j] = corr
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.isnan(corr_matrix)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=task_names, yticklabels=task_names, mask=mask,
                square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
    
    plt.title('Correlation Matrix Between Tox21 Tasks\n(Based on overlapping valid samples)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure if requested
    if save_figures:
        filename = 'task_correlations.png'
        filepath = os.path.join('./Figures', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved: {filepath}")
    
    plt.show()
    
    # Find highest correlations
    print("Strongest correlations between tasks:")
    print("=" * 40)
    correlations = []
    for i in range(n_tasks):
        for j in range(i+1, n_tasks):
            if not np.isnan(corr_matrix[i, j]):
                correlations.append((task_names[i], task_names[j], corr_matrix[i, j]))
    
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    for task1, task2, corr in correlations[:10]:
        print(f"{task1:15} ↔ {task2:15}: {corr:6.3f}")

# Analyze task correlations
analyze_task_correlations(all_labels, task_names)

#%%
def analyze_missing_data_patterns(all_labels, task_names, save_figures=True):
    """Visualize missing data patterns"""
    import os
    from datetime import datetime
    
    print("\n" + "="*60)
    print("MISSING DATA PATTERN ANALYSIS")
    print("="*60)
    
    # Create figures directory if it doesn't exist
    if save_figures:
        os.makedirs('./Figures', exist_ok=True)
    
    missing_matrix = np.isnan(all_labels)
    
    # Overall statistics
    total_values = missing_matrix.size
    missing_values = np.sum(missing_matrix)
    print(f"Total missing values: {missing_values:,} / {total_values:,} ({missing_values/total_values*100:.1f}%)")
    
    # Missing per task
    print("\nMissing data per task:")
    for i, task_name in enumerate(task_names):
        task_missing = np.sum(missing_matrix[:, i])
        task_total = len(all_labels)
        print(f"{task_name:15}: {task_missing:4d} / {task_total:4d} ({task_missing/task_total*100:5.1f}%)")
    
    # Missing per sample
    missing_per_sample = np.sum(missing_matrix, axis=1)
    print(f"\nSamples with 0 missing: {np.sum(missing_per_sample == 0):,}")
    print(f"Samples with all missing: {np.sum(missing_per_sample == len(task_names)):,}")
    
    # Visualize missing pattern
    plt.figure(figsize=(14, 8))
    
    # Sample random subset for visualization
    sample_size = min(1000, len(all_labels))
    sample_indices = np.random.choice(len(all_labels), sample_size, replace=False)
    sample_indices.sort()
    
    plt.imshow(missing_matrix[sample_indices].T, cmap='RdYlBu', aspect='auto', interpolation='nearest')
    plt.xlabel('Sample Index')
    plt.ylabel('Task')
    plt.title(f'Missing Data Pattern (Random sample of {sample_size} molecules)\nRed = Missing, Blue = Present')
    plt.yticks(range(len(task_names)), task_names)
    plt.colorbar(label='Missing (1) / Present (0)')
    plt.tight_layout()
    
    # Save figure if requested
    if save_figures:
        filename = f'missing_data_patterns.png'
        filepath = os.path.join('./Figures', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved: {filepath}")
    
    plt.show()

# Analyze missing data patterns
analyze_missing_data_patterns(all_labels, task_names)

#%%
def print_summary():
    """Print final summary"""
    print("\n" + "="*80)
    print("TOX21 DATASET VISUALIZATION COMPLETE!")
    print("="*80)
    print("Key insights:")
    print("- 12 different toxicity assays with varying class imbalances")
    print("- Significant missing data across all tasks")
    print("- Molecular graphs show structural diversity")
    print("- Some tasks are correlated, suggesting related biological pathways")
    print("- Ready for machine learning model development!")
    print("\nNext steps for your project:")
    print("1. Choose 1-2 specific tasks for focused modeling")
    print("2. Implement GNN baseline (GCN/GAT)")
    print("3. Add XGBoost comparison on molecular fingerprints")
    print("4. Implement explainability (SHAP, attention)")
    print("5. Connect predictions to known toxicophores")

# Print summary
print_summary()

#%%
# Quick data access for further analysis
print("Data variables available for further analysis:")
print(f"- dataset: {type(dataset)} with {len(dataset)} molecules")
print(f"- all_labels: {all_labels.shape} numpy array")
print(f"- task_names: {len(task_names)} tasks")
print(f"- stats_df: DataFrame with task statistics")
print("\nExample usage:")
print("# Focus on NR-AhR task (liver toxicity)")
print("focus_task = 2")
print("task_labels = all_labels[:, focus_task]")
print("valid_mask = ~np.isnan(task_labels)")
print("print(f'Valid samples for {task_names[focus_task]}: {np.sum(valid_mask)}')")