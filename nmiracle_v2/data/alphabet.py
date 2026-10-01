import os
import numpy as np
import lmdb
import pickle
import zlib
from nmiracle.data.tokenizer import BasicSmilesTokenizer
from typing import Set, Optional
from omegaconf import DictConfig
from pathlib import Path


def generate_alphabet(
    config: DictConfig,
    output_path: Optional[str] = None,
    tokenizer: Optional[BasicSmilesTokenizer] = None,
) -> np.ndarray:
    """Generate alphabet from the appropriate dataset based on config"""
    
    if tokenizer is None:
        tokenizer = BasicSmilesTokenizer()
    
    print("Generating comprehensive SMILES alphabet...")
    
    unique_tokens = set()
    
    # 🆕 DETECT DATASET TYPE AND USE APPROPRIATE SOURCE
    dataset_name = getattr(config, 'dataset_name', 'alberts')
        
    if dataset_name == 'alberts_dataset':
        # Use Alberts LMDB database
        training_stage = getattr(config, 'training_stage')

        if training_stage == 'sub2struct':
            dataset_path = Path(config.sub2struct.data_dir) / 'combined_pretrain_smiles.npy'
            print(f"Collecting tokens from NMR2Struct pretrain + Alberts: {dataset_path}")

            smiles_array = np.load(dataset_path, allow_pickle=True)

            for smiles_bytes in smiles_array:
                smiles = smiles_bytes.decode('utf-8') if isinstance(smiles_bytes, bytes) else smiles_bytes
                tokens = tokenizer.tokenize(smiles)
                unique_tokens.update(tokens)

            print(f"Added {len(unique_tokens)} unique tokens from NMR2Struct pretrain + Alberts datasets")

        elif training_stage == 'spec2struct':
            dataset_path = Path(config.spec2struct.data_dir) / 'combined_pretrain_smiles.npy'
            print(f"Collecting tokens from Alberts LMDB: {dataset_path}")
            try:
                env = lmdb.open(dataset_path, readonly=True, lock=False)
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for key, value in cursor:
                        if key != b'__len__':
                            sample = pickle.loads(zlib.decompress(value))
                            if 'smiles' in sample:
                                tokens = tokenizer.tokenize(sample['smiles'])
                                unique_tokens.update(tokens)
                env.close()
                
                print(f"Added {len(unique_tokens)} unique tokens from Alberts dataset")
            except Exception as e:
                print(f"Error processing Alberts dataset: {e}")
        else:
            print(f"Warning: Alberts LMDB not found at {dataset_path}")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_name}")
    
    # Create sorted alphabet array
    alphabet = np.array(sorted(list(unique_tokens)))
    
    print(f"Generated alphabet with {len(alphabet)} tokens")
    
    # Save alphabet if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, alphabet)
        print(f"Saved alphabet to {output_path}")
    
    return alphabet


def _get_tokens_from_smiles_file(smiles_path: str, tokenizer: BasicSmilesTokenizer) -> Set[str]:
    """Extract unique tokens from a file containing SMILES strings"""
    unique_tokens = set()
    try:
        # Try to load with numpy first
        try:
            smiles_list = np.load(smiles_path, allow_pickle=True)
        except:
            # If that fails, try pickle
            with open(smiles_path, 'rb') as f:
                smiles_list = pickle.load(f)
                
        for smiles in smiles_list:
            tokens = tokenizer.tokenize(smiles)
            unique_tokens.update(tokens)
    except Exception as e:
        print(f"Error processing SMILES file: {e}")
    
    return unique_tokens

def generate_or_load_alphabet(
    config: DictConfig,
    alphabet_path: Optional[str] = None,
    tokenizer: Optional[BasicSmilesTokenizer] = None,
) -> np.ndarray:
  
    # Check if we should use existing alphabet
    if alphabet_path and os.path.exists(alphabet_path):
        print(f"Loading existing alphabet from {alphabet_path}")
        return np.load(alphabet_path)

    cfg = config.data if hasattr(config, 'data') else config
    return generate_alphabet(cfg, alphabet_path, tokenizer)