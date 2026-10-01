from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import Levenshtein
from rdkit import Chem, DataStructs
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import MACCSkeys
import os
# from myopic_mces import MCES

def are_constitutional_isomers(smi1, smi2):
    """Check if two SMILES are identical ignoring stereochemistry."""
    if smi1 is None or smi2 is None:
        return False
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    can1 = Chem.MolToSmiles(mol1, isomericSmiles=False)
    can2 = Chem.MolToSmiles(mol2, isomericSmiles=False)
    return can1 == can2

def are_enantiomers(smi1, smi2):
    if smi1 is None or smi2 is None:
        return False
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    can_smi1 = Chem.MolToSmiles(mol1)
    can_smi2 = Chem.MolToSmiles(mol2)
    return ("@" in can_smi1 and "@" in can_smi2
        and can_smi1.replace(
            "@@", "__DOUBLE_AT__"
        ).replace("@", "@@").replace(
            "__DOUBLE_AT__", "@"
        ) == can_smi2)

def molecules_match_enantiomer_aware(pred_smiles, true_smiles):
    """
    Check if prediction matches ground truth (allowing enantiomers).
    Returns True if exact match OR enantiomers.
    Assumes input SMILES are already canonical.
    """
    # Exact match OR enantiomer match
    return (pred_smiles == true_smiles) or are_enantiomers(pred_smiles, true_smiles)


def calculate_rdkit_similarity(smiles1, smiles2):
    """Compute similarity between molecules using RDKit fingerprints."""
    
    if isinstance(smiles1, Chem.Mol) and isinstance(smiles2, Chem.Mol):
        mol1 = smiles1
        mol2 = smiles2
    else:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = RDKFingerprint(mol1)
    fp2 = RDKFingerprint(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)

def calculate_maccs_similarity(smiles1, smiles2):
    """Compute similarity between molecules using MACCS keys."""    
    if isinstance(smiles1, Chem.Mol) and isinstance(smiles2, Chem.Mol):
        mol1 = smiles1
        mol2 = smiles2
    else:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0

    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def is_valid_smiles(smiles):
    """Check if SMILES is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def selfies_to_smiles(selfies_string):
    """Decode a SELFIES string to SMILES."""
    import selfies as sf
    return sf.decoder(selfies_string)


def is_valid_molecule(molecule_string, model_format):
    """Check if string is valid in the given format"""
    if model_format.lower() == "selfies":
        try:
            smiles = selfies_to_smiles(molecule_string)
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    else:  # SMILES
        mol = Chem.MolFromSmiles(molecule_string)
        return mol is not None

# def compute_mces_distance(smiles1, smiles2):
#     """Compute MCES distance using myopic_mces"""
#     if smiles1 is None or smiles2 is None:
#         return float('inf')
#     result = MCES(smiles1, smiles2)
#     return result[1]
# def compute_mces_distance(smiles1, smiles2):
#     """Compute MCES distance using myopic_mces, silencing CBC solver output."""
#     if smiles1 is None or smiles2 is None:
#         return float('inf')
#     with open(os.devnull, 'w') as devnull:
#         old_fds = os.dup(1), os.dup(2)
#         os.dup2(devnull.fileno(), 1)
#         os.dup2(devnull.fileno(), 2)
#         try:
#             result = MCES(smiles1, smiles2)
#         finally:
#             os.dup2(old_fds[0], 1)
#             os.dup2(old_fds[1], 2)
#             os.close(old_fds[0])
#             os.close(old_fds[1])
#     return result[1]
                
def compute_levenshtein_distance(smiles1, smiles2):
    """Computes Levenshtein distance between two SMILES strings."""
    if smiles1 is None or smiles2 is None:
        return float('inf')
    return Levenshtein.distance(smiles1, smiles2)


def calculate_tanimoto_similarity(gen_smiles, true_smiles, use_chirality=False):
    """Compute Tanimoto similarity between generated and true SMILES."""
    if gen_smiles is None or true_smiles is None:
        return 0.0
    if isinstance(gen_smiles, Chem.Mol) and isinstance(true_smiles, Chem.Mol):
        gen_mol = gen_smiles
        true_mol = true_smiles
    else:
        gen_mol = Chem.MolFromSmiles(gen_smiles)
        true_mol = Chem.MolFromSmiles(true_smiles)
    if gen_mol and true_mol:
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, radius=2, nBits=2048, useChirality=use_chirality)
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, radius=2, nBits=2048, useChirality=use_chirality)
        return DataStructs.TanimotoSimilarity(gen_fp, true_fp)
    return 0.0  # If invalid molecule, return 0

def compute_molecular_weight(smiles):
    """Computes molecular weight for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Descriptors.MolWt(mol) if mol else float('inf')  # Handle invalid molecules

def canonicalize_smiles(smiles):
    """Canonicalizes a SMILES string using RDKit."""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None