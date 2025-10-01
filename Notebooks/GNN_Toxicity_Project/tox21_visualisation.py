#%%
"""
Tox21 Dataset Visualization Script - LIVER TOXICITY FOCUSED
Based on the structure: 7823 molecules, 12 tasks, labels shape (1, 12)
Focus: NR-AhR task (Aryl hydrocarbon receptor - liver toxicity)
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
    print(f"✓ FOCUS: NR-AhR (index 2) - Aryl hydrocarbon receptor (liver toxicity)")
    
    return dataset, all_labels, task_names

# Load data
dataset, all_labels, task_names = load_and_prepare_data()

#%%
def analyze_liver_task_only(all_labels, task_names, save_figures=True):
    """Analyze and visualize class distribution for NR-AhR (liver toxicity) task only"""
    import os
    
    print("\n" + "="*60)
    print("LIVER TOXICITY (NR-AhR) CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Create figures directory if it doesn't exist
    if save_figures:
        os.makedirs('./Figures', exist_ok=True)
    
    # Focus on NR-AhR task (liver toxicity, index 2)
    focus_task_idx = 2
    focus_task_name = task_names[focus_task_idx]
    task_labels = all_labels[:, focus_task_idx]
    
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
            'Task': focus_task_name,
            'Total': total_samples,
            'Valid': valid_count,
            'Missing': missing_count,
            'Non-hepatotoxic (0)': inactive_count,
            'Hepatotoxic (1)': active_count,
            'Other': other_count,
            'Missing %': (missing_count / total_samples) * 100,
            'Hepatotoxic %': (active_count / valid_count) * 100 if valid_count > 0 else 0
        }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'NR-AhR (Liver Toxicity) Task - Class Distribution', fontsize=16)
        
        # Bar plot
        counts = [inactive_count, active_count, missing_count]
        labels = ['Non-hepatotoxic', 'Hepatotoxic', 'Missing']
        colors = ['lightgreen', 'red', 'lightgray']
        
        bars = ax1.bar(labels, counts, color=colors, alpha=0.8)
        ax1.set_title(f'Sample Distribution\n(n={valid_count} valid samples)', fontsize=12)
        ax1.set_ylabel('Number of Molecules', fontsize=11)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Pie chart for valid labels only
        if active_count > 0 and inactive_count > 0:
            valid_counts = [inactive_count, active_count]
            valid_labels_pie = ['Non-hepatotoxic', 'Hepatotoxic']
            colors_pie = ['lightgreen', 'red']
            
            wedges, texts, autotexts = ax2.pie(valid_counts, 
                                             labels=valid_labels_pie,
                                             colors=colors_pie,
                                             autopct='%1.1f%%',
                                             startangle=90)
            ax2.set_title(f'Valid Labels Distribution\n({valid_count:,} molecules)')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_figures:
            filename = f'liver_toxicity_distribution.png'
            filepath = os.path.join('./Figures', filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Figure saved: {filepath}")
        
        plt.show()
        
        # Print detailed statistics
        print(f"\nNR-AhR (Liver Toxicity) Statistics:")
        print("=" * 50)
        print(f"Total molecules: {total_samples:,}")
        print(f"Valid labels: {valid_count:,} ({valid_count/total_samples*100:.1f}%)")
        print(f"Missing labels: {missing_count:,} ({missing_count/total_samples*100:.1f}%)")
        print(f"Non-hepatotoxic: {inactive_count:,} ({inactive_count/valid_count*100:.1f}% of valid)")
        print(f"Hepatotoxic: {active_count:,} ({active_count/valid_count*100:.1f}% of valid)")
        print(f"Class imbalance ratio: {inactive_count/active_count:.1f}:1 (non-toxic:toxic)")
        
        return pd.DataFrame([stats])

# Analyze liver task only
liver_stats_df = analyze_liver_task_only(all_labels, task_names)


#%%
def print_liver_focused_summary():
    """Print liver-focused summary"""
    print("\n" + "="*80)
    print("LIVER TOXICITY (NR-AhR) ANALYSIS COMPLETE!")
    print("="*80)
    print("Key insights for hepatotoxicity prediction:")
    
    # Get liver task statistics
    liver_task_idx = 2
    liver_labels = all_labels[:, liver_task_idx]
    valid_mask = ~np.isnan(liver_labels)
    valid_count = np.sum(valid_mask)
    hepatotoxic_count = np.sum(liver_labels == 1)
    
    print(f"📊 Dataset: {valid_count:,} molecules with valid liver toxicity labels")
    print(f"📊 Class balance: {hepatotoxic_count:,} hepatotoxic ({hepatotoxic_count/valid_count*100:.1f}%) vs {valid_count-hepatotoxic_count:,} non-hepatotoxic")
    print(f"📊 Imbalance ratio: {(valid_count-hepatotoxic_count)/hepatotoxic_count:.1f}:1 (non-toxic:toxic)")
    
    print(f"\n🎯 Focus for ML modeling:")
    print(f"   • Primary task: NR-AhR (Aryl hydrocarbon receptor)")
    print(f"   • Biological relevance: Liver metabolism, hepatotoxicity, DILI prediction") 
    print(f"   • Data quality: {valid_count/len(all_labels)*100:.1f}% complete labels")
    print(f"   • Challenge: Class imbalance typical of toxicity datasets")
    
    print(f"\n🧪 Next steps for GNN vs LGBM comparison:")
    print(f"   1. ✅ Data understanding complete")
    print(f"   2. 🎯 Split data: {valid_count} samples → train/val/test")
    print(f"   3. 🏗️  Build LGBM baseline with liver-relevant molecular descriptors")
    print(f"   4. 🧠 Build GNN model on molecular graphs")
    print(f"   5. 📈 Compare performance and interpretability")
    print(f"   6. 🔍 Connect predictions to known hepatotoxic structural alerts")

# Print liver-focused summary
print_liver_focused_summary()

#%%
# Quick data access for liver toxicity modeling
liver_task_idx = 2
liver_labels = all_labels[:, liver_task_idx]
valid_mask = ~np.isnan(liver_labels)

print("Data ready for liver toxicity modeling:")
print(f"- dataset: {type(dataset)} with {len(dataset)} molecules")
print(f"- liver_labels: {liver_labels.shape} array (NR-AhR task)")
print(f"- valid_mask: {np.sum(valid_mask):,} molecules with valid liver toxicity labels")
print(f"- liver_props_df: DataFrame with {len(liver_props_df) if 'liver_props_df' in locals() else 0} analyzed molecules")

print(f"\nReady to build models:")
print(f"# Extract liver toxicity data")
print(f"liver_molecules = [dataset[i] for i in range(len(dataset)) if valid_mask[i]]")
print(f"liver_targets = liver_labels[valid_mask]")
print(f"print(f'Ready for modeling: {{len(liver_molecules)}} molecules')")