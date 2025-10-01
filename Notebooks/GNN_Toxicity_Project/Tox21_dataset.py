"""
Tox21 Dataset Exploration Script
Comprehensive analysis and visualization of the Tox21 toxicity prediction dataset
"""

#%%
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
from torch_geometric.transforms import Compose


#%%
# RDKit imports for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available. Install with: conda install -c rdkit rdkit")
    RDKIT_AVAILABLE = False

# NetworkX for graph visualization
import networkx as nx

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_tox21_dataset():
    """Load and inspect the Tox21 dataset"""
    print("Loading Tox21 dataset...")
    dataset = MoleculeNet(root='./data', name='Tox21')
    
    print(f"Dataset: {dataset}")
    print(f"Number of molecules: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}")
    
    return dataset

def inspect_data_structure(dataset):
    """Inspect the structure of individual data points"""
    print("\n" + "="*50)
    print("DATA STRUCTURE ANALYSIS")
    print("="*50)
    
    # Look at first molecule
    data = dataset[0]
    print(f"Sample molecule structure:")
    print(f"- Node features shape: {data.x.shape}")
    print(f"- Edge indices shape: {data.edge_index.shape}")
    print(f"- Edge attributes shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
    print(f"- Labels shape: {data.y.shape}")
    print(f"- SMILES available: {hasattr(data, 'smiles')}")
    
    # Check for missing labels
    print(f"\nLabel analysis for first molecule:")
    print(f"Labels: {data.y}")
    print(f"Missing labels (NaN): {torch.isnan(data.y).sum().item()}")
    
    return data

def get_tox21_task_names():
    """Get the names of the 12 Tox21 tasks"""
    task_names = [
        'NR-AR',      # Androgen receptor
        'NR-AR-LBD',  # Androgen receptor ligand binding domain
        'NR-AhR',     # Aryl hydrocarbon receptor (liver toxicity related)
        'NR-Aromatase', # Aromatase
        'NR-ER',      # Estrogen receptor
        'NR-ER-LBD',  # Estrogen receptor ligand binding domain
        'NR-PPAR-gamma', # Peroxisome proliferator-activated receptor gamma
        'SR-ARE',     # Antioxidant response element
        'SR-ATAD5',   # ATPase family AAA domain-containing protein 5
        'SR-HSE',     # Heat shock factor response element
        'SR-MMP',     # Mitochondrial membrane potential
        'SR-p53'      # p53 signaling pathway
    ]
    return task_names

def analyze_class_distribution(dataset):
    """Analyze the distribution of toxicity classes across all tasks"""
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    
    task_names = get_tox21_task_names()
    
    # Collect all labels
    all_labels = []
    for i in range(len(dataset)):
        all_labels.append(dataset[i].y.numpy())
    
    labels_array = np.array(all_labels)
    
    # Create subplots for all tasks
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Tox21 Dataset: Class Distribution Across All Tasks', fontsize=16)
    
    axes = axes.flatten()
    
    overall_stats = []
    
    for task_idx, task_name in enumerate(task_names):
        task_labels = labels_array[:, task_idx]
        
        # Count classes (0: non-toxic, 1: toxic, NaN: missing)
        valid_mask = ~np.isnan(task_labels)
        valid_labels = task_labels[valid_mask]
        
        if len(valid_labels) > 0:
            toxic_count = np.sum(valid_labels == 1)
            non_toxic_count = np.sum(valid_labels == 0)
            missing_count = np.sum(np.isnan(task_labels))
            
            # Plot histogram
            ax = axes[task_idx]
            counts = [non_toxic_count, toxic_count, missing_count]
            labels = ['Non-toxic', 'Toxic', 'Missing']
            colors = ['lightblue', 'salmon', 'lightgray']
            
            bars = ax.bar(labels, counts, color=colors)
            ax.set_title(f'{task_name}\n(n={len(valid_labels)} valid)')
            ax.set_ylabel('Count')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                           str(count), ha='center', va='bottom')
            
            # Store stats
            overall_stats.append({
                'Task': task_name,
                'Non-toxic': non_toxic_count,
                'Toxic': toxic_count,
                'Missing': missing_count,
                'Total_valid': len(valid_labels),
                'Toxic_ratio': toxic_count / len(valid_labels) if len(valid_labels) > 0 else 0
            })
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    stats_df = pd.DataFrame(overall_stats)
    print("\nSummary Statistics:")
    print(stats_df.to_string(index=False))
    
    return stats_df

def visualize_molecular_graphs(dataset, num_samples=6):
    """Visualize molecular graphs using NetworkX"""
    print("\n" + "="*50)
    print("MOLECULAR GRAPH VISUALIZATION")
    print("="*50)
    
    # Find examples for NR-AhR task (index 2 - liver toxicity related)
    task_idx = 2  # NR-AhR
    task_name = get_tox21_task_names()[task_idx]
    
    toxic_examples = []
    non_toxic_examples = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        label = data.y[task_idx]
        
        if not torch.isnan(label):
            if label == 1 and len(toxic_examples) < num_samples // 2:
                toxic_examples.append(i)
            elif label == 0 and len(non_toxic_examples) < num_samples // 2:
                non_toxic_examples.append(i)
        
        if len(toxic_examples) >= num_samples // 2 and len(non_toxic_examples) >= num_samples // 2:
            break
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 8))
    fig.suptitle(f'Molecular Graphs - {task_name} Task', fontsize=16)
    
    # Plot toxic examples
    for i, mol_idx in enumerate(toxic_examples):
        data = dataset[mol_idx]
        G = to_networkx(data, to_undirected=True)
        
        ax = axes[0, i]
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, node_color='salmon', node_size=100, 
               with_labels=False, edge_color='gray', alpha=0.7)
        ax.set_title(f'Toxic #{mol_idx}\n({G.number_of_nodes()} atoms)')
        ax.axis('off')
    
    # Plot non-toxic examples
    for i, mol_idx in enumerate(non_toxic_examples):
        data = dataset[mol_idx]
        G = to_networkx(data, to_undirected=True)
        
        ax = axes[1, i]
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, node_color='lightblue', node_size=100,
               with_labels=False, edge_color='gray', alpha=0.7)
        ax.set_title(f'Non-toxic #{mol_idx}\n({G.number_of_nodes()} atoms)')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_molecular_structures_rdkit(dataset, num_samples=6):
    """Visualize actual molecular structures using RDKit (if available)"""
    if not RDKIT_AVAILABLE:
        print("RDKit not available. Skipping molecular structure visualization.")
        return
    
    print("\n" + "="*50)
    print("MOLECULAR STRUCTURE VISUALIZATION (RDKit)")
    print("="*50)
    
    # Find examples for NR-AhR task
    task_idx = 2
    task_name = get_tox21_task_names()[task_idx]
    
    toxic_examples = []
    non_toxic_examples = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        label = data.y[task_idx]
        
        if not torch.isnan(label) and hasattr(data, 'smiles'):
            if label == 1 and len(toxic_examples) < num_samples // 2:
                toxic_examples.append((i, data.smiles))
            elif label == 0 and len(non_toxic_examples) < num_samples // 2:
                non_toxic_examples.append((i, data.smiles))
        
        if len(toxic_examples) >= num_samples // 2 and len(non_toxic_examples) >= num_samples // 2:
            break
    
    # Create molecular images
    all_mols = []
    all_legends = []
    
    for mol_idx, smiles in toxic_examples:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            all_mols.append(mol)
            all_legends.append(f'Toxic #{mol_idx}')
    
    for mol_idx, smiles in non_toxic_examples:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            all_mols.append(mol)
            all_legends.append(f'Non-toxic #{mol_idx}')
    
    if all_mols:
        img = Draw.MolsToGridImage(all_mols, molsPerRow=num_samples//2, 
                                  subImgSize=(300, 300), legends=all_legends)
        # Save and display
        img.save('tox21_molecular_structures.png')
        print(f"Molecular structures saved as 'tox21_molecular_structures.png'")
        print(f"Showing {len(all_mols)} molecules for {task_name} task")

def analyze_molecular_properties(dataset):
    """Analyze molecular properties using RDKit"""
    if not RDKIT_AVAILABLE:
        print("RDKit not available. Skipping molecular property analysis.")
        return
    
    print("\n" + "="*50)
    print("MOLECULAR PROPERTY ANALYSIS")
    print("="*50)
    
    properties = []
    labels_list = []
    
    for i in range(min(1000, len(dataset))):  # Sample first 1000 for speed
        data = dataset[i]
        if hasattr(data, 'smiles'):
            mol = Chem.MolFromSmiles(data.smiles)
            if mol:
                props = {
                    'MolWt': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'NumHDonors': Descriptors.NumHDonors(mol),
                    'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                    'NumAromaticRings': Descriptors.NumAromaticRings(mol)
                }
                properties.append(props)
                labels_list.append(data.y.numpy())
    
    if not properties:
        print("No valid molecular structures found.")
        return
    
    # Create DataFrame
    props_df = pd.DataFrame(properties)
    labels_array = np.array(labels_list)
    
    # Plot property distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle('Molecular Property Distributions', fontsize=16)
    axes = axes.flatten()
    
    for i, prop in enumerate(props_df.columns):
        if i < len(axes):
            ax = axes[i]
            ax.hist(props_df[prop], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel(prop)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{prop} Distribution')
    
    # Remove empty subplot
    if len(props_df.columns) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nMolecular Property Summary:")
    print(props_df.describe())

def analyze_task_correlations(dataset):
    """Analyze correlations between different toxicity tasks"""
    print("\n" + "="*50)
    print("TASK CORRELATION ANALYSIS")
    print("="*50)
    
    # Collect labels for all molecules
    all_labels = []
    for i in range(len(dataset)):
        all_labels.append(dataset[i].y.numpy())
    
    labels_array = np.array(all_labels)
    task_names = get_tox21_task_names()
    
    # Calculate correlation matrix (only for non-missing values)
    corr_matrix = np.full((len(task_names), len(task_names)), np.nan)
    
    for i in range(len(task_names)):
        for j in range(len(task_names)):
            task_i = labels_array[:, i]
            task_j = labels_array[:, j]
            
            # Find molecules with valid labels for both tasks
            valid_mask = ~(np.isnan(task_i) | np.isnan(task_j))
            
            if np.sum(valid_mask) > 10:  # Need at least 10 samples
                corr = np.corrcoef(task_i[valid_mask], task_j[valid_mask])[0, 1]
                corr_matrix[i, j] = corr
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.isnan(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=task_names, yticklabels=task_names, mask=mask,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix Between Tox21 Tasks')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("="*60)
    print("TOX21 DATASET EXPLORATION")
    print("="*60)
    
    # Load dataset
    dataset = load_tox21_dataset()
    
    # Inspect data structure
    sample_data = inspect_data_structure(dataset)
    
    # Analyze class distributions
    stats_df = analyze_class_distribution(dataset)
    
    # Visualize molecular graphs
    visualize_molecular_graphs(dataset)
    
    # Visualize molecular structures (if RDKit available)
    visualize_molecular_structures_rdkit(dataset)
    
    # Analyze molecular properties
    analyze_molecular_properties(dataset)
    
    # Analyze task correlations
    analyze_task_correlations(dataset)
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE!")
    print("="*60)
    print("Key findings:")
    print("- Dataset contains", len(dataset), "molecules")
    print("- 12 different toxicity tasks")
    print("- Significant class imbalance in most tasks")
    print("- Missing labels are common (real-world challenge)")
    print("- Tasks show varying degrees of correlation")

if __name__ == "__main__":
    main()
# %%
