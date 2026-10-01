import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# RDKit imports for molecule processing and visualization
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")
sns.set_palette("colorblind")

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate and visualize molecular generation results')
parser.add_argument('--visualize-only', default=False,
                   help='Skip metric calculation and only visualize from saved metrics')
parser.add_argument('--metrics-file', type=str, default="specmolt5_v2/eval_results/all_model_metrics.pkl",
                   help='Path to saved metrics file')
parser.add_argument('--sample-size', type=int, default=None,
                   help='Sample size for debugging (set to None to use all data)')
args = parser.parse_args()

# Define paths to test result files from different models
result_files = {
    # "SpecMolFusion(IR + HNMR + CNMR) (70-10-20 split, seed 42)": "/Users/federico/Desktop/imperial/my_multimodal/specmolt5_v2/ckpts/specmolfusion-ir_hnmr_cnmr-bs-64-lr-0.00005-formula-false/test_results/test_molecules.pkl",
    # "SpecMolFusion(IR + HNMR + CNMR + formula) (90-10 split, seed 3345)": "/Users/federico/Desktop/imperial/my_multimodal/specmolt5_v2/ckpts/specmolfusion-ir_hnmr_cnmr-bs-64-lr-0.00005-formula-true-90_10/test_results/test_molecules.pkl",
    # "SpecMolFusion(IR + HNMR + CNMR + formula) (70-10-20 split, seed 42)": "/Users/federico/Desktop/imperial/my_multimodal/specmolt5_v2/ckpts/specmolfusion-ir_hnmr_cnmr-bs-64-lr-0.00005-formula-true-70_10_20/test_results/test_molecules.pkl",
    "Alberts (HNMR + formula)": "/Users/federico/Desktop/imperial/my_multimodal/alberts_baseline/runs/runs_f_groups/h_nmr/data/test_molecules.pkl",
    # Add other models when available, e.g.:
    # "SpecMolFusion_IR_Only_formula": "../specmolt5_v2/ckpts/specmolfusion-ir_hnmr_cnmr-bs-64-lr-0.00005-formula-false/test_results/test_molecules.pkl",
    # "DiffMS_IR_Only": "../diffms_v2/test_results/ir_only/test_molecules.pkl",
    # "DiffMS_Full": "../diffms_v2/test_results/full_spectra/test_molecules.pkl",
}

# Function to load pickle files
def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Calculate common molecular metrics
def calculate_metrics(true_smiles, generated_smiles_list):
    """
    Calculate metrics between a true SMILES and a list of generated SMILES
    Returns the best metrics across all generated molecules
    """
    if not generated_smiles_list:  # If no valid molecules were generated
        return {
            "valid": False,
            "tanimoto": 0.0,
            "rdkit_fp_sim": 0.0,
            "maccs_sim": 0.0,
            # "mces_distance": 1.0,  # Maximum distance for no valid molecules
            "exact_match": False,
            "num_valid": 0
        }
    
    # Create RDKit molecule for the true SMILES
    true_mol = Chem.MolFromSmiles(true_smiles)
    
    # Calculate Morgan fingerprint for true molecule
    true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
    true_rdkit_fp = Chem.RDKFingerprint(true_mol)
    true_maccs = AllChem.GetMACCSKeysFingerprint(true_mol)
    
    # Initialize best metrics
    best_tanimoto = 0.0
    best_rdkit_sim = 0.0
    best_maccs_sim = 0.0
    # best_mces_distance = 1.0  # Start with maximum distance
    exact_match = False
    
    # Check each generated molecule
    for gen_smiles in generated_smiles_list:
        # Check for exact match
        if gen_smiles == true_smiles:
            exact_match = True
            best_tanimoto = 1.0
            best_rdkit_sim = 1.0
            best_maccs_sim = 1.0
            # best_mces_distance = 0.0  # Zero distance for identical molecules
            break
            
        # Create RDKit molecule for generated SMILES
        gen_mol = Chem.MolFromSmiles(gen_smiles)
        
        # Calculate fingerprints for generated molecule
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
        gen_rdkit_fp = Chem.RDKFingerprint(gen_mol)
        gen_maccs = AllChem.GetMACCSKeysFingerprint(gen_mol)
        
        # Calculate similarities
        tanimoto = TanimotoSimilarity(true_fp, gen_fp)
        rdkit_sim = TanimotoSimilarity(true_rdkit_fp, gen_rdkit_fp)
        maccs_sim = TanimotoSimilarity(true_maccs, gen_maccs)
        # mces_distance = compute_mces_distance(true_smiles, gen_smiles)

        # Update best metrics
        best_tanimoto = max(best_tanimoto, tanimoto)
        best_rdkit_sim = max(best_rdkit_sim, rdkit_sim)
        best_maccs_sim = max(best_maccs_sim, maccs_sim)
        # best_mces_distance = min(best_mces_distance, mces_distance)  # We want minimum distance

    return {
        "valid": True,
        "tanimoto": best_tanimoto,
        "rdkit_fp_sim": best_rdkit_sim,
        "maccs_sim": best_maccs_sim,
        # "mces_distance": best_mces_distance,  # Add the MCES distance
        "exact_match": exact_match,
        "num_valid": len(generated_smiles_list)
    }

# Compute comprehensive metrics for all models
def compute_model_metrics(model_results):
    model_metrics = {}
    
    for model_name, results in model_results.items():
        print(f"Computing metrics for {model_name}...")
        true_smiles = results['all_canon_true']
        generated_lists = results['all_canon_generated']
        total_molecules = len(true_smiles)
        
        # Initialize metrics storage
        all_metrics = []
        formula_matches = 0

        # Calculate metrics for each molecule
        for i in tqdm(range(total_molecules), desc=model_name):
            metrics = calculate_metrics(true_smiles[i], generated_lists[i])
            all_metrics.append(metrics)

            # Calculate formula accuracy
            if calculate_formula_accuracy(true_smiles[i], generated_lists[i]):
                formula_matches += 1
        

        # Calculate formula accuracy rate
        formula_accuracy = formula_matches / total_molecules if total_molecules > 0 else 0

        # Calculate summary statistics
        valid_generations = sum(1 for metrics in all_metrics if metrics['valid'])
        valid_rate = valid_generations / total_molecules if total_molecules > 0 else 0
        
        exact_matches = sum(1 for metrics in all_metrics if metrics['exact_match'])
        exact_match_rate = exact_matches / total_molecules if total_molecules > 0 else 0
        
        # Calculate mean similarity metrics (only for valid generations)
        valid_metrics = [m for m in all_metrics if m['valid']]
        
        mean_tanimoto = np.mean([m['tanimoto'] for m in valid_metrics]) if valid_metrics else 0
        mean_rdkit_sim = np.mean([m['rdkit_fp_sim'] for m in valid_metrics]) if valid_metrics else 0
        mean_maccs_sim = np.mean([m['maccs_sim'] for m in valid_metrics]) if valid_metrics else 0
        
        # Calculate top-k accuracy
        # A molecule is in top-k if any of the k first generated molecules exactly match the true molecule
        top_k_counts = {1: 0, 3: 0, 5: 0, 10: 0}
        
        for i, gen_list in enumerate(generated_lists):
            true_mol = true_smiles[i]
            for k in top_k_counts.keys():
                top_k_mols = gen_list[:k] if len(gen_list) >= k else gen_list
                if true_mol in top_k_mols:
                    top_k_counts[k] += 1
        
        top_k_accuracy = {k: count/total_molecules for k, count in top_k_counts.items()}
        
        # Store metrics (add formula accuracy)
        model_metrics[model_name] = {
            'total_molecules': total_molecules,
            'valid_rate': valid_rate,
            'exact_match_rate': exact_match_rate,
            'formula_accuracy': formula_accuracy,  # New metric
            'mean_tanimoto': mean_tanimoto,
            'mean_rdkit_sim': mean_rdkit_sim,
            'mean_maccs_sim': mean_maccs_sim,
            'top_k_accuracy': top_k_accuracy,
            'all_metrics': all_metrics
        }
    
    return model_metrics

# Save the calculated metrics for reuse
def save_metrics(all_model_metrics, filename="all_model_metrics.pkl"):
    """Save computed metrics to disk for later reuse"""
    with open(filename, 'wb') as f:
        pickle.dump(all_model_metrics, f)
    print(f"Metrics saved to {filename}")

def load_saved_metrics(filename="all_model_metrics.pkl"):
    """Load previously computed metrics from disk"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Metrics file {filename} not found!")
        return None

# Create comparative visualizations
def plot_comparison_bar(df, metrics, title, figsize=(12, 6)):
    """Create a grouped bar plot comparing multiple metrics across models"""
    plt.figure(figsize=figsize)
    
    # Set width of bars
    barWidth = 0.15
    positions = np.arange(len(df))
    
    # Plot bars
    for i, metric in enumerate(metrics):
        plt.bar(positions + i*barWidth, df[metric], width=barWidth, label=metric)
    
    # Add labels and legend
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(positions + barWidth * (len(metrics)-1)/2, df['Model'], rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    
    return plt

# Distribution analysis of similarity scores
def plot_similarity_distributions(all_model_metrics):
    """Plot distributions of similarity scores for each model"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Prepare data for each model
    for model_name, metrics in all_model_metrics.items():
        # Extract valid metrics
        valid_metrics = [m for m in metrics['all_metrics'] if m['valid']]
        
        # Extract similarity values
        tanimoto_values = [m['tanimoto'] for m in valid_metrics]
        rdkit_values = [m['rdkit_fp_sim'] for m in valid_metrics]
        maccs_values = [m['maccs_sim'] for m in valid_metrics]
        
        # Plot distributions
        sns.kdeplot(tanimoto_values, ax=axs[0], label=model_name)
        sns.kdeplot(rdkit_values, ax=axs[1], label=model_name)
        sns.kdeplot(maccs_values, ax=axs[2], label=model_name)
    
    # Set titles and labels
    axs[0].set_title('Morgan Fingerprint Similarity')
    axs[0].set_xlabel('Tanimoto Similarity')
    axs[0].set_ylabel('Density')
    
    axs[1].set_title('RDKit Fingerprint Similarity')
    axs[1].set_xlabel('Tanimoto Similarity')
    axs[1].set_ylabel('Density')
    
    axs[2].set_title('MACCS Keys Similarity')
    axs[2].set_xlabel('Tanimoto Similarity')
    axs[2].set_ylabel('Density')
    
    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def calculate_formula_accuracy(true_smiles, generated_smiles_list):
    """
    Determines if any generated molecule has the same molecular formula as the true molecule
    """
    if not generated_smiles_list:  # If no valid molecules were generated
        return False
    
    # Get formula from true SMILES
    true_mol = Chem.MolFromSmiles(true_smiles)
    true_formula = rdMolDescriptors.CalcMolFormula(true_mol)
    
    # Check each generated molecule
    for gen_smiles in generated_smiles_list:
        try:
            gen_mol = Chem.MolFromSmiles(gen_smiles)
            if gen_mol:
                gen_formula = rdMolDescriptors.CalcMolFormula(gen_mol)
                if gen_formula == true_formula:
                    return True
        except:
            continue
    
    return False

def visualize_molecule_examples(model_results, num_examples=15, focus_on="all"):
    """
    Visualize example molecules from different models with customizable sampling
    
    Args:
        model_results: Dictionary containing model results
        num_examples: Number of examples to visualize per quality group
        focus_on: Which quality group to focus on ("high", "medium", "low", or "all")
    """
    # Create output directory if it doesn't exist
    os.makedirs("specmolt5_v2/eval_results/molecule_examples", exist_ok=True)
    
    # Get the model names and select models
    selected_models = list(model_results.keys())
    
    # Sample molecules of different quality based on tanimoto similarity
    for model_name in selected_models:
        print(f"Creating visualizations for model: {model_name}")
        
        # Get true and generated molecules
        true_smiles = model_results[model_name]['all_canon_true']
        generated_lists = model_results[model_name]['all_canon_generated']
        
        # Calculate tanimoto similarities for all molecules
        similarities = []
        valid_indices = []
        
        for idx, (true, gen_list) in enumerate(zip(true_smiles, generated_lists)):
            if not gen_list:  # Skip molecules with no valid generations
                continue
                
            # Compute similarity of best molecule
            true_mol = Chem.MolFromSmiles(true)
            best_sim = 0
            
            for gen_smiles in gen_list:
                gen_mol = Chem.MolFromSmiles(gen_smiles)
                if gen_mol:
                    true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
                    gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
                    sim = TanimotoSimilarity(true_fp, gen_fp)
                    best_sim = max(best_sim, sim)
            
            similarities.append(best_sim)
            valid_indices.append(idx)
        
        # Sort by similarity (pairs of similarity value and molecule index)
        sorted_pairs = sorted(zip(similarities, valid_indices), key=lambda x: x[0])
        
        # Select examples based on focus parameter
        if len(sorted_pairs) == 0:
            print(f"No valid molecules found for {model_name}")
            continue
            
        # Determine how to select examples based on the focus
        if focus_on == "all":
            # Select examples evenly from low, medium and high similarity
            num_per_group = max(1, num_examples // 3)
            
            # Low similarity examples
            low_start = 0
            low_indices = [p[1] for p in sorted_pairs[low_start:low_start+num_per_group]]
            
            # Medium similarity examples
            med_start = len(sorted_pairs) // 2 - num_per_group // 2
            med_indices = [p[1] for p in sorted_pairs[med_start:med_start+num_per_group]]
            
            # High similarity examples
            high_start = max(0, len(sorted_pairs) - num_per_group)
            high_indices = [p[1] for p in sorted_pairs[high_start:]]
            
            quality_groups = [
                ("low", low_indices),
                ("medium", med_indices),
                ("high", high_indices)
            ]
            
        elif focus_on == "high":
            # Focus on high similarity examples
            high_start = max(0, len(sorted_pairs) - num_examples)
            high_indices = [p[1] for p in sorted_pairs[high_start:]]
            quality_groups = [("high", high_indices)]
            
        elif focus_on == "medium":
            # Focus on medium similarity examples
            med_start = len(sorted_pairs) // 2 - num_examples // 2
            med_indices = [p[1] for p in sorted_pairs[med_start:med_start+num_examples]]
            quality_groups = [("medium", med_indices)]
            
        elif focus_on == "low":
            # Focus on low similarity examples
            low_indices = [p[1] for p in sorted_pairs[:num_examples]]
            quality_groups = [("low", low_indices)]
            
        else:
            # Default to evenly distributed
            num_per_group = max(1, num_examples // 3)
            total_groups = min(3, len(sorted_pairs))
            group_size = len(sorted_pairs) // total_groups
            
            quality_groups = []
            for i in range(total_groups):
                start_idx = i * group_size
                end_idx = start_idx + min(num_per_group, group_size)
                if i == total_groups - 1:  # Last group gets any remaining examples
                    end_idx = len(sorted_pairs)
                    
                quality_name = "low" if i == 0 else "high" if i == total_groups - 1 else "medium"
                indices = [p[1] for p in sorted_pairs[start_idx:end_idx]]
                quality_groups.append((quality_name, indices))
        
        # Visualize each quality group
        for quality, indices in quality_groups:
            for counter, idx in enumerate(indices):
                # Get similarity value for this example
                sim_value = next(sim for sim, i in sorted_pairs if i == idx)
                
                # Get true molecule
                true_smiles_i = true_smiles[idx]
                true_mol = Chem.MolFromSmiles(true_smiles_i)
                
                # Create list of molecules to visualize
                mols_to_draw = [true_mol]
                legends = ['True Molecule']
                
                # Add top 3 generated molecules (or fewer if we don't have 3)
                generated_list = generated_lists[idx][:3]
                
                # Find tanimoto similarities for each
                similarities = []
                for gen_smiles in generated_list:
                    gen_mol = Chem.MolFromSmiles(gen_smiles)
                    true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
                    gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
                    sim = TanimotoSimilarity(true_fp, gen_fp)
                    similarities.append(sim)
                    mols_to_draw.append(gen_mol)
                    legends.append(f"Candidate {len(mols_to_draw)-1} (sim={sim:.3f})")
                
                # Create molecule grid
                img = Draw.MolsToGridImage(
                    mols_to_draw,
                    molsPerRow=2,  # 2 molecules per row for better layout
                    subImgSize=(300, 300),
                    legends=legends
                )
                
                # Save the image
                filename = f"specmolt5_v2/eval_results/molecule_examples/{model_name}_{quality}_similarity_{counter}_{idx}.png"
                img.save(filename)
                print(f"  Saved {quality} quality example {counter+1}/{len(indices)} (idx={idx}) with similarity {sim_value:.3f}")
                
                # Print SMILES for reference
                print(f"  True SMILES: {true_smiles_i}")
                print(f"  Gen SMILES: {generated_list[0] if generated_list else 'None'}\n")


# Analyze molecular properties
def calculate_molecular_properties(smiles):
    """Calculate basic molecular properties for a SMILES string"""
    if not smiles:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'Rings': Descriptors.RingCount(mol),
        'HeavyAtoms': mol.GetNumHeavyAtoms(),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol)
    }

# Sample molecules for property analysis
def analyze_molecular_properties(model_results, sample_size=None):
    """Compare molecular properties between true and generated molecules"""
    # Initialize dataframe for property analysis
    property_data = []
    
    # Select a random sample of molecules for analysis
    first_model = next(iter(model_results.values()))
    total_molecules = len(first_model['all_canon_true'])

    if sample_size is None:
        sample_size = total_molecules
        sample_indices = range(total_molecules)
    else:
        sample_indices = np.random.choice(total_molecules, min(sample_size, total_molecules), replace=False)
    
    # For each model, analyze properties
    for model_name, results in model_results.items():
        true_smiles = results['all_canon_true']
        generated_lists = results['all_canon_generated']
        
        for idx in sample_indices:
            # Get true molecule properties
            true_props = calculate_molecular_properties(true_smiles[idx])
            if true_props:
                true_props['SMILES'] = true_smiles[idx]
                true_props['Model'] = model_name
                true_props['Type'] = 'True'
                property_data.append(true_props)
            
            # Get first generated molecule properties (if any valid ones exist)
            if generated_lists[idx]:
                gen_props = calculate_molecular_properties(generated_lists[idx][0])
                if gen_props:
                    gen_props['SMILES'] = generated_lists[idx][0]
                    gen_props['Model'] = model_name
                    gen_props['Type'] = 'Generated'
                    property_data.append(gen_props)
    
    # Create dataframe
    props_df = pd.DataFrame(property_data)
    return props_df

# Plot property distributions
def plot_property_distributions(props_df):
    """Plot distributions of molecular properties"""
    properties = ['MW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'Rings', 'TPSA']
    
    # Calculate number of rows/columns for subplot grid
    n_props = len(properties)
    n_cols = 3
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, prop in enumerate(properties):
        ax = axes[i]
        for model_name in props_df['Model'].unique():
            model_df = props_df[props_df['Model'] == model_name]
            
            # Plot true and generated distributions
            sns.kdeplot(
                data=model_df, 
                x=prop, 
                hue='Type',
                ax=ax,
                label=f"{model_name} - {{}}".format("True/Generated")
            )
        
        ax.set_title(f'{prop} Distribution')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Turn off any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

# Statistical analysis of property matching
def property_correlation_analysis(props_df):
    """Analyze how well generated molecules match properties of true molecules"""
    # Initialize results
    correlation_data = []
    
    # For each model, calculate property correlations
    for model_name in props_df['Model'].unique():
        model_df = props_df[props_df['Model'] == model_name]
        
        # Group by molecule type
        true_df = model_df[model_df['Type'] == 'True']
        gen_df = model_df[model_df['Type'] == 'Generated']
        
        # Properties to analyze
        properties = ['MW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'Rings', 'TPSA', 'FractionCSP3', 'HeavyAtoms']
        
        # Calculate correlations for each property
        for prop in properties:
            if prop in true_df.columns and prop in gen_df.columns:
                # Get values
                true_values = true_df[prop].values
                gen_values = gen_df[prop].values
                
                # Calculate correlation if we have enough data points
                if len(true_values) > 5 and len(gen_values) > 5:
                    # Make sure arrays are same length (use minimum)
                    min_len = min(len(true_values), len(gen_values))
                    true_values = true_values[:min_len]
                    gen_values = gen_values[:min_len]
                    
                    correlation = np.corrcoef(true_values, gen_values)[0, 1]
                    
                    correlation_data.append({
                        'Model': model_name,
                        'Property': prop,
                        'Correlation': correlation
                    })
    
    # Create dataframe
    corr_df = pd.DataFrame(correlation_data)
    
    # Plot correlations
    plt.figure(figsize=(12, 6))
    sns.barplot(data=corr_df, x='Property', y='Correlation', hue='Model')
    plt.title('Property Correlation between True and Generated Molecules')
    plt.xlabel('Molecular Property')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return corr_df

# Main execution logic
if __name__ == "__main__":
    # Load model results
    model_results = {}
    for model_name, filepath in result_files.items():
        if os.path.exists(filepath):
            print(f"Loading results for {model_name}...")
            model_results[model_name] = load_results(filepath)
        else:
            print(f"Warning: File not found for {model_name} at {filepath}")
    
    # For debugging, limit the data size if sample_size is provided
    if args.sample_size:
        for model_name in model_results:
            model_results[model_name]['all_canon_true'] = model_results[model_name]['all_canon_true'][:args.sample_size]
            model_results[model_name]['all_canon_generated'] = model_results[model_name]['all_canon_generated'][:args.sample_size]
    
    if args.visualize_only:
        # Load pre-calculated metrics
        all_model_metrics = load_saved_metrics(args.metrics_file)
        if all_model_metrics is None:
            print("Error: Cannot visualize without metrics. Run without --visualize-only to calculate metrics first.")
            exit(1)
    else:
        # Compute metrics for all models
        #debugghino <3
        # model_results['SpecMolFusion(IR + HNMR + CNMR)']['all_canon_true'] = model_results['SpecMolFusion(IR + HNMR + CNMR)']['all_canon_true'][:1000]
        # model_results['SpecMolFusion(IR + HNMR + CNMR)']['all_canon_generated'] = model_results['SpecMolFusion(IR + HNMR + CNMR)']['all_canon_generated'][:1000]
        all_model_metrics = compute_model_metrics(model_results)
        # Save calculated metrics
        save_metrics(all_model_metrics, args.metrics_file)
    
    # Create summary dataframe
    summary_data = []
    for model_name, metrics in all_model_metrics.items():
        model_row = {
            'Model': model_name,
            'Total Molecules': metrics['total_molecules'],
            'Valid Rate': metrics['valid_rate'],
            'Formula Accuracy': metrics['formula_accuracy'],  # New metric
            'Exact Match Rate': metrics['exact_match_rate'],
            'Mean Tanimoto': metrics['mean_tanimoto'],
            'Mean RDKit Sim': metrics['mean_rdkit_sim'],
            'Mean MACCS Sim': metrics['mean_maccs_sim'],
            'Top-1 Accuracy': metrics['top_k_accuracy'][1],
            'Top-3 Accuracy': metrics['top_k_accuracy'][3],
            'Top-5 Accuracy': metrics['top_k_accuracy'][5],
            'Top-10 Accuracy': metrics['top_k_accuracy'][10]}
        summary_data.append(model_row)
    
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary of results:")
    print(summary_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    # Plot similarity metrics
    similarity_metrics = ['Mean Tanimoto', 'Mean RDKit Sim', 'Mean MACCS Sim']
    plot_comparison_bar(summary_df, similarity_metrics, 'Molecular Similarity Comparison')
    plt.savefig('similarity_comparison.png')
    plt.close()
    
    # Plot accuracy metrics
    accuracy_metrics = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Top-10 Accuracy']
    plot_comparison_bar(summary_df, accuracy_metrics, 'Top-K Accuracy Comparison')
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    # Plot validity, formula accuracy, and exact match rate
    basic_metrics = ['Valid Rate', 'Formula Accuracy', 'Exact Match Rate'] 
    plot_comparison_bar(summary_df, basic_metrics, 'Validity, Formula Accuracy and Exact Match Rate')
    plt.savefig('validity_comparison.png')
    plt.close()
    
    # Plot similarity distributions
    similarity_fig = plot_similarity_distributions(all_model_metrics)
    plt.savefig('similarity_distributions.png')
    plt.close()
    
    # Analyze molecular properties
    print("\nAnalyzing molecular properties...")
    props_df = analyze_molecular_properties(model_results, args.sample_size)
    
    # Molecular example visualization
    # print("\nVisualizing example molecules...")
    # visualize_molecule_examples(model_results, num_examples=15) #5 per group qualit (high, medium, low)

    # Plot property distributions
    # property_fig = plot_property_distributions(props_df)
    # plt.savefig('property_distributions.png')
    # plt.close()
    
    # Analyze property correlations
    property_corr_df = property_correlation_analysis(props_df)
    plt.savefig('property_correlations.png')
    plt.close()
    
    # Save results to CSV for further analysis
    print("\nSaving results to CSV...")
    summary_df.to_csv("specmolt5_v2/eval_results/model_comparison_summary.csv", index=False)
    props_df.to_csv("specmolt5_v2/eval_results/molecular_properties_comparison.csv", index=False)
    
    print("\nEvaluation complete!")