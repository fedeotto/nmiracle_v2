"""
This script recomputes ALL metrics with three levels of stereochemistry handling:
1. Exact match (strict)
2. Enantiomer-tolerant match (appropriate for NMR)
3. Constitution-only match (no stereochemistry)

Includes: Tanimoto, RDKit, MACCS, Levenshtein, MCES, exact match, and top-k accuracies
"""
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from common.eval_utils import (compute_mces_distance, 
                               calculate_tanimoto_similarity, 
                               calculate_rdkit_similarity, calculate_maccs_similarity, 
                               compute_levenshtein_distance,
                               are_enantiomers)
from rdkit import Chem
import os
import numpy as np

# Silencing RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def process_molecule(args):
    """Process a single molecule - designed for multiprocessing."""
    i, gen_smiles_list, true_smiles = args
    
    result = {
        'stereo': {'has_stereo': False, 'single': False, 'multiple': False, 'unassigned': False, 'multiple_unassigned': False},
        'valid': False,
        'exact': {'found': False, 'rank': -1, 'tanimoto': 0, 'rdkit': 0, 'maccs': 0, 'levenshtein': float('inf')},
        'enantiomer': {'found': False, 'rank': -1, 'tanimoto': 0, 'rdkit': 0, 'maccs': 0, 'levenshtein': float('inf')},
        'constitution': {'found': False, 'rank': -1, 'tanimoto': 0, 'rdkit': 0, 'maccs': 0, 'levenshtein': float('inf'), 'mces': float('inf')},
    }
    
    # Stereochemistry stats
    num_chiral, num_unassigned = count_stereocenters(true_smiles)
    if num_chiral > 0:
        result['stereo']['has_stereo'] = True
        result['stereo']['single'] = (num_chiral == 1)
        result['stereo']['multiple'] = (num_chiral > 1)
        if num_unassigned > 0:
            result['stereo']['unassigned'] = True
            result['stereo']['multiple_unassigned'] = (num_chiral > 1)
    
    if not gen_smiles_list:
        return result
    
    result['valid'] = True
    
    mol_true = Chem.MolFromSmiles(true_smiles)
    if mol_true is None:
        return result
    
    can_true_with_stereo = Chem.MolToSmiles(mol_true)
    can_true_no_stereo = Chem.MolToSmiles(mol_true, isomericSmiles=False)
    
    for rank, pred_smiles in enumerate(gen_smiles_list):
        try:
            mol_pred = Chem.MolFromSmiles(pred_smiles)
        except Exception as e:
             print(e)
             mol_pred = None
             
        if mol_pred is None:
            continue
        
        can_pred_with_stereo = Chem.MolToSmiles(mol_pred)
        can_pred_no_stereo = Chem.MolToSmiles(mol_pred, isomericSmiles=False)
        
        exact_match = (can_pred_with_stereo == can_true_with_stereo)
        enantiomer_match = exact_match or are_enantiomers(can_pred_with_stereo, can_true_with_stereo)
        constitution_match = (can_pred_no_stereo == can_true_no_stereo)
        
        if exact_match and not result['exact']['found']:
            result['exact']['found'] = True
            result['exact']['rank'] = rank
        if enantiomer_match and not result['enantiomer']['found']:
            result['enantiomer']['found'] = True
            result['enantiomer']['rank'] = rank
        if constitution_match and not result['constitution']['found']:
            result['constitution']['found'] = True
            result['constitution']['rank'] = rank
        
        # Similarities with stereo
        tanimoto = calculate_tanimoto_similarity(can_pred_with_stereo, can_true_with_stereo, use_chirality=True)
        rdkit_sim = calculate_rdkit_similarity(can_pred_with_stereo, can_true_with_stereo)
        maccs_sim = calculate_maccs_similarity(can_pred_with_stereo, can_true_with_stereo)
        levenshtein = compute_levenshtein_distance(can_pred_with_stereo, can_true_with_stereo)
        
        if tanimoto > result['exact']['tanimoto']:
            result['exact']['tanimoto'] = tanimoto
        if rdkit_sim > result['exact']['rdkit']:
            result['exact']['rdkit'] = rdkit_sim
        if maccs_sim > result['exact']['maccs']:
            result['exact']['maccs'] = maccs_sim
        if levenshtein < result['exact']['levenshtein']:
            result['exact']['levenshtein'] = levenshtein
        # Enantiomer level (same fingerprints)
        if tanimoto > result['enantiomer']['tanimoto']:
            result['enantiomer']['tanimoto'] = tanimoto
        if rdkit_sim > result['enantiomer']['rdkit']:
            result['enantiomer']['rdkit'] = rdkit_sim
        if maccs_sim > result['enantiomer']['maccs']:
            result['enantiomer']['maccs'] = maccs_sim
        if levenshtein < result['enantiomer']['levenshtein']:
            result['enantiomer']['levenshtein'] = levenshtein
        
        # Constitutional (no stereo)
        tanimoto_const = calculate_tanimoto_similarity(can_pred_no_stereo, can_true_no_stereo, use_chirality=False)
        rdkit_const = calculate_rdkit_similarity(can_pred_no_stereo, can_true_no_stereo)
        maccs_const = calculate_maccs_similarity(can_pred_no_stereo, can_true_no_stereo)
        levenshtein_const = compute_levenshtein_distance(can_pred_no_stereo, can_true_no_stereo)
        # mces_const = compute_mces_distance(can_pred_no_stereo, can_true_no_stereo)
        
        if tanimoto_const > result['constitution']['tanimoto']:
            result['constitution']['tanimoto'] = tanimoto_const
        if rdkit_const > result['constitution']['rdkit']:
            result['constitution']['rdkit'] = rdkit_const
        if maccs_const > result['constitution']['maccs']:
            result['constitution']['maccs'] = maccs_const
        if levenshtein_const < result['constitution']['levenshtein']:
            result['constitution']['levenshtein'] = levenshtein_const
        # if mces_const < result['constitution']['mces']:
        #     result['constitution']['mces'] = mces_const
    
    return result

def count_stereocenters(smiles):
    """Count chiral centers (assigned and unassigned) and stereogenic bonds"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0, 0, 0
    
    # Find chiral centers (including unassigned)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    num_chiral_total = len(chiral_centers)
    num_chiral_unassigned = sum(1 for _, label in chiral_centers if label == '?')
    # Focus only on Chirality
    # Count stereogenic double bonds
    # num_stereo_bonds = 0
    # for bond in mol.GetBonds():
    #     if bond.GetBondType() == Chem.BondType.DOUBLE:
    #         if bond.GetStereo() != Chem.BondStereo.STEREONONE:
    #             num_stereo_bonds += 1
    
    return num_chiral_total, num_chiral_unassigned

def molecules_match_enantiomer_aware(pred_smiles, true_smiles):
    """
    Check if molecules match considering enantiomers as equivalent.
    Returns: (exact_match, enantiomer_match, constitution_match)
    """
    mol_pred = Chem.MolFromSmiles(pred_smiles)
    mol_true = Chem.MolFromSmiles(true_smiles)
    
    if mol_pred is None or mol_true is None:
        return False, False, False
    
    # Canonicalize both
    can_pred = Chem.MolToSmiles(mol_pred)
    can_true = Chem.MolToSmiles(mol_true)
    
    # 1. Exact match
    exact_match = (can_pred == can_true)
    
    # 2. Enantiomer match
    enantiomer_match = exact_match or are_enantiomers(can_pred, can_true)
    
    # 3. Constitutional match (no stereochemistry)
    can_pred_no_stereo = Chem.MolToSmiles(mol_pred, isomericSmiles=False)
    can_true_no_stereo = Chem.MolToSmiles(mol_true, isomericSmiles=False)
    constitution_match = (can_pred_no_stereo == can_true_no_stereo)
    
    return exact_match, enantiomer_match, constitution_match

if __name__ == "__main__":
    results_path = 'important_ckpts/spec2struct-alberts-subvocab-ir-hnmr-cnmr-cnn'
    results_name = 'test_results-temperature-1.0-top_k-5_seed-2.json'

    n_workers= min(4, cpu_count())  # Limit to 16 CPUs to avoid overloading
    with open(os.path.join(results_path, results_name), 'r') as f:
        results = json.load(f)

    results = results['generations']
    all_canon_true = results['all_canon_true']
    all_canon_generated = results['all_canon_generated']
    total_molecules = len(all_canon_true)
    print(f"Processing {total_molecules} molecules with {cpu_count()} CPUs...")

    # Prepare args for parallel processing
    args_list = [(i, gen, true) for i, (gen, true) in enumerate(zip(all_canon_generated, all_canon_true))]

    # Parallel execution with progress logging
    results_list = []    
    with Pool(n_workers) as pool:
        for i, result in enumerate(tqdm(pool.imap(process_molecule, args_list), 
                                        total=total_molecules, 
                                        desc="Computing metrics")):
            results_list.append(result)

    # Aggregate results
    stereo_stats = {'total': 0, 'has_stereo': 0, 'single_stereocenter': 0, 'multiple_stereocenters': 0, 'unassigned_stereo': 0, 'multiple_with_unassigned': 0}
    exact_top_1 = exact_top_3 = exact_top_5 = exact_top_10 = exact_top_15 = exact_match_count = 0
    enantiomer_top_1 = enantiomer_top_3 = enantiomer_top_5 = enantiomer_top_10 = enantiomer_top_15 = enantiomer_match_count = 0
    constitution_top_1 = constitution_top_3 = constitution_top_5 = constitution_top_10 = constitution_top_15 = constitution_match_count = 0
    exact_tanimoto, exact_rdkit, exact_maccs, exact_levenshtein = [], [], [], []
    enantiomer_tanimoto, enantiomer_rdkit, enantiomer_maccs, enantiomer_levenshtein = [], [], [], []
    constitution_tanimoto, constitution_rdkit, constitution_maccs, constitution_levenshtein, constitution_mces = [], [], [], [], []
    valid_count = 0

    for r in results_list:
        stereo_stats['total'] += 1
        if r['stereo']['has_stereo']:
            stereo_stats['has_stereo'] += 1
            if r['stereo']['single']:
                stereo_stats['single_stereocenter'] += 1
            if r['stereo']['multiple']:
                stereo_stats['multiple_stereocenters'] += 1
            if r['stereo']['unassigned']:
                stereo_stats['unassigned_stereo'] += 1
            if r['stereo']['multiple_unassigned']:
                stereo_stats['multiple_with_unassigned'] += 1
        
        if not r['valid']:
            continue
        valid_count += 1
        
        # Exact
        exact_tanimoto.append(r['exact']['tanimoto'])
        exact_rdkit.append(r['exact']['rdkit'])
        exact_maccs.append(r['exact']['maccs'])
        exact_levenshtein.append(r['exact']['levenshtein'])
        if r['exact']['found']:
            exact_match_count += 1
            if r['exact']['rank'] < 1: exact_top_1 += 1
            if r['exact']['rank'] < 3: exact_top_3 += 1
            if r['exact']['rank'] < 5: exact_top_5 += 1
            if r['exact']['rank'] < 10: exact_top_10 += 1
            if r['exact']['rank'] < 15: exact_top_15 += 1
        
        # Enantiomer
        enantiomer_tanimoto.append(r['enantiomer']['tanimoto'])
        enantiomer_rdkit.append(r['enantiomer']['rdkit'])
        enantiomer_maccs.append(r['enantiomer']['maccs'])
        enantiomer_levenshtein.append(r['enantiomer']['levenshtein'])
        if r['enantiomer']['found']:
            enantiomer_match_count += 1
            if r['enantiomer']['rank'] < 1: enantiomer_top_1 += 1
            if r['enantiomer']['rank'] < 3: enantiomer_top_3 += 1
            if r['enantiomer']['rank'] < 5: enantiomer_top_5 += 1
            if r['enantiomer']['rank'] < 10: enantiomer_top_10 += 1
            if r['enantiomer']['rank'] < 15: enantiomer_top_15 += 1
        
        # Constitution
        constitution_tanimoto.append(r['constitution']['tanimoto'])
        constitution_rdkit.append(r['constitution']['rdkit'])
        constitution_maccs.append(r['constitution']['maccs'])
        constitution_levenshtein.append(r['constitution']['levenshtein'])
        # constitution_mces.append(r['constitution']['mces'])
        if r['constitution']['found']:
            constitution_match_count += 1
            if r['constitution']['rank'] < 1: constitution_top_1 += 1
            if r['constitution']['rank'] < 3: constitution_top_3 += 1
            if r['constitution']['rank'] < 5: constitution_top_5 += 1
            if r['constitution']['rank'] < 10: constitution_top_10 += 1
            if r['constitution']['rank'] < 15: constitution_top_15 += 1

    # Calculate final metrics
    print("\n" + "="*80)
    print("STEREOCHEMISTRY STATISTICS IN TEST SET")
    print("="*80)
    print(f"Total molecules: {stereo_stats['total']}")
    print(f"Molecules with stereochemistry: {stereo_stats['has_stereo']} ({stereo_stats['has_stereo']/stereo_stats['total']*100:.1f}%)")
    print(f"  - Single stereocenter: {stereo_stats['single_stereocenter']}")
    print(f"  - Multiple stereocenters: {stereo_stats['multiple_stereocenters']}")
    print(f"Molecules with unassigned stereocenters: {stereo_stats['unassigned_stereo']}")
    print(f"  - Multiple centers with unassigned: {stereo_stats['multiple_with_unassigned']}")

    print("\n" + "="*80)
    print("EVALUATION RESULTS - ALL METRICS")
    print("="*80)

    print("\n1. EXACT MATCH (Strict - all stereochemistry must match)")
    print("-" * 80)
    print(f"Exact match rate:      {exact_match_count/total_molecules:.4f}")
    print(f"Top-1 accuracy:        {exact_top_1/total_molecules:.4f}")
    print(f"Top-3 accuracy:        {exact_top_3/total_molecules:.4f}")
    print(f"Top-5 accuracy:        {exact_top_5/total_molecules:.4f}")
    print(f"Tanimoto similarity:   {sum(exact_tanimoto)/len(exact_tanimoto):.4f}")
    print(f"RDKit similarity:      {sum(exact_rdkit)/len(exact_rdkit):.4f}")
    print(f"MACCS similarity:      {sum(exact_maccs)/len(exact_maccs):.4f}")
    print(f"Levenshtein distance:  {sum(exact_levenshtein)/len(exact_levenshtein):.4f}")

    print("\n2. ENANTIOMER-AWARE MATCH (NMR-appropriate - treats enantiomers as equivalent)")
    print("-" * 80)
    print(f"Match rate:            {enantiomer_match_count/total_molecules:.4f}")
    print(f"Top-1 accuracy:        {enantiomer_top_1/total_molecules:.4f}")
    print(f"Top-3 accuracy:        {enantiomer_top_3/total_molecules:.4f}")
    print(f"Top-5 accuracy:        {enantiomer_top_5/total_molecules:.4f}")
    print(f"Top-10 accuracy:       {enantiomer_top_10/total_molecules:.4f}")
    print(f"Top-15 accuracy:       {enantiomer_top_15/total_molecules:.4f}")
    print(f"Tanimoto similarity:   {sum(enantiomer_tanimoto)/len(enantiomer_tanimoto):.4f}")
    print(f"RDKit similarity:      {sum(enantiomer_rdkit)/len(enantiomer_rdkit):.4f}")
    print(f"MACCS similarity:      {sum(enantiomer_maccs)/len(enantiomer_maccs):.4f}")
    print(f"Levenshtein distance:  {sum(enantiomer_levenshtein)/len(enantiomer_levenshtein):.4f}")

    print("\n3. CONSTITUTIONAL MATCH (No stereochemistry)")
    print("-" * 80)
    print(f"Match rate:            {constitution_match_count/total_molecules:.4f}")
    print(f"Top-1 accuracy:        {constitution_top_1/total_molecules:.4f}")
    print(f"Top-3 accuracy:        {constitution_top_3/total_molecules:.4f}")
    print(f"Top-5 accuracy:        {constitution_top_5/total_molecules:.4f}")
    print(f"Top-10 accuracy:       {constitution_top_10/total_molecules:.4f}")
    print(f"Top-15 accuracy:       {constitution_top_15/total_molecules:.4f}")
    print(f"Tanimoto similarity:   {sum(constitution_tanimoto)/len(constitution_tanimoto):.4f}")
    print(f"RDKit similarity:      {sum(constitution_rdkit)/len(constitution_rdkit):.4f}")
    print(f"MACCS similarity:      {sum(constitution_maccs)/len(constitution_maccs):.4f}")
    # print(f"MCES distance:       {sum(constitution_mces)/len(constitution_mces):.4f}")
    print(f"Levenshtein distance:  {sum(constitution_levenshtein)/len(constitution_levenshtein):.4f}")

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("- Exact match: Original baseline comparison (overly strict for NMR)")
    print("- Enantiomer-aware: Scientifically appropriate for NMR-based prediction")
    print("- Constitutional: Tests molecular connectivity only")
    print("\nRECOMMENDATION: Use 'Enantiomer-aware' as primary metric in paper")
    print("="*80)

    # Save results
    results_detailed = {
        'stereochemistry_stats': stereo_stats,
        'metrics': {
            'exact_match': {
                'exact_match_rate': exact_match_count/total_molecules,
                'top_1_accuracy': exact_top_1/total_molecules,
                'top_3_accuracy': exact_top_3/total_molecules,
                'top_5_accuracy': exact_top_5/total_molecules,
                'top_10_accuracy': exact_top_10/total_molecules,
                'top_15_accuracy': exact_top_15/total_molecules,
                'tanimoto_similarity_mean': np.mean(exact_tanimoto),
                'tanimoto__similarity_std': np.std(exact_tanimoto),
                'rdkit_similarity_mean': np.mean(exact_rdkit),
                'rdkit_similarity_std': np.std(exact_rdkit),
                'maccs_similarity_mean': np.mean(exact_maccs),
                'maccs_similarity_std': np.std(exact_maccs),
                'levenshtein_distance_mean': np.mean(exact_levenshtein),
                'levenshtein_distance_std': np.std(exact_levenshtein),
            },
            'enantiomer_aware': {
                'match_rate': enantiomer_match_count/total_molecules,
                'top_1_accuracy': enantiomer_top_1/total_molecules,
                'top_3_accuracy': enantiomer_top_3/total_molecules,
                'top_5_accuracy': enantiomer_top_5/total_molecules,
                'top_10_accuracy': enantiomer_top_10/total_molecules,
                'top_15_accuracy': enantiomer_top_15/total_molecules,
                'tanimoto_similarity_mean': np.mean(enantiomer_tanimoto),
                'tanimoto_similarity_std': np.std(enantiomer_tanimoto),
                'rdkit_similarity_mean': np.mean(enantiomer_rdkit),
                'rdkit_similarity_std': np.std(enantiomer_rdkit),
                'maccs_similarity_mean': np.mean(enantiomer_maccs),
                'maccs_similarity_std': np.std(enantiomer_maccs),
                'levenshtein_distance_mean': np.mean(enantiomer_levenshtein),
                'levenshtein_distance_std': np.std(enantiomer_levenshtein),
            },
            'constitutional': {
                'match_rate': constitution_match_count/total_molecules,
                'top_1_accuracy': constitution_top_1/total_molecules,
                'top_3_accuracy': constitution_top_3/total_molecules,
                'top_5_accuracy': constitution_top_5/total_molecules,
                'top_10_accuracy': constitution_top_10/total_molecules,
                'top_15_accuracy': constitution_top_15/total_molecules,
                'tanimoto_similarity_mean': np.mean(constitution_tanimoto),
                'tanimoto_similarity_std': np.std(constitution_tanimoto),
                'rdkit_similarity_mean': np.mean(constitution_rdkit),
                'rdkit_similarity_std': np.std(constitution_rdkit),
                'maccs_similarity_mean': np.mean(constitution_maccs),
                'maccs_similarity_std': np.std(constitution_maccs),
                # 'mces_distance_mean ': np.mean(constitution_mces),
                # 'mces_distance_std ': np.std(constitution_mces),
                'levenshtein_distance': np.mean(constitution_levenshtein),
                'levenshtein_distance_std': np.std(constitution_levenshtein)
            }
        }
    }

    output_path = os.path.join(results_path, f'enantiomer_aware_all_metrics_{results_name}')
    with open(output_path, 'w') as f:
        json.dump(results_detailed, f, indent=4)

    print(f"\nResults saved to: {output_path}")


# Exact match:
# Top-k accuracy is the fraction of molecules where the ground truth appears in the top-k predictions with all stereochemistry matching.
# Enantiomer-aware:
# Top-k accuracy is the fraction where the ground truth or its enantiomer appears in the top-k predictions. This is appropriate for NMR, which cannot distinguish enantiomers.
# Constitutional:
# Top-k accuracy is the fraction where the ground truth appears in the top-k predictions ignoring all stereochemistry (just connectivity).
