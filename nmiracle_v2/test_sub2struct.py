import json
import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from nmiracle_v2.data.datamodule import SpectralDataModule
from nmiracle_v2.data.tokenizer import BasicSmilesTokenizer
from nmiracle_v2.configuration import ExperimentConfig
from nmiracle_v2.models.pl_module import Sub2StructModule
from nmiracle_v2.utils.generation import generate_canonical_smiles, move_to_device

os.environ["HYDRA_FULL_ERROR"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "sub2struct-subvocab"
ALPHABET_PATH = 'nmiracle_v2/utils/smiles_alphabet.npy'
CKPT_NAME = "last.ckpt"
TEMPERATURE = 1.0
TOP_K = 5 #for top-k sampling
NUM_SEQUENCES = 15
SEED = 0
MAX_MOLECULES = 100000
BATCH_SIZE = 8

def parse_arguments():
    parser = ArgumentParser(description="Test a Sub2Struct model on the pre-training set.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model directory containing the checkpoint.")
    parser.add_argument("--ckpt_name", type=str, default=CKPT_NAME, help="Name of the checkpoint file to load.")
    parser.add_argument("--alphabet_path", type=str, default=ALPHABET_PATH, help="Path to the alphabet .npy file for the tokenizer.")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_model_and_checkpoint(model_path, ckpt_name):
    ckpt_path = Path(model_path) / ckpt_name
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ExperimentConfig.from_dict(checkpoint["hyper_parameters"])
    return config, checkpoint


def main():
    args = parse_arguments()
    config, checkpoint = load_model_and_checkpoint(args.model_path, args.ckpt_name)
    pl.seed_everything(args.seed)

    tokenizer = BasicSmilesTokenizer()
    tokenizer.setup_alphabet(np.load(args.alphabet_path, allow_pickle=True))

    pl_module = Sub2StructModule(config=config, tokenizer=tokenizer).to(DEVICE)
    pl_module.load_state_dict(checkpoint["state_dict"], strict=True)
    pl_module.eval()
    print("Model state loaded from checkpoint.✅")

    #Setting custom batch size
    config.data.dataset.bs_test = BATCH_SIZE
    data_module = SpectralDataModule(config=config, tokenizer=tokenizer)
    data_module.setup(stage="test")

    generations = {
        "all_canon_true": [],
        "all_canon_generated": [],
    }
    max_molecule_len = config.data.dataset.max_molecule_len

    for batch in tqdm(data_module.test_dataloader()):
        batch = move_to_device(batch, DEVICE)

        # Keep gradients enabled for the Transformer encoder forward pass.
        with torch.enable_grad():
            outputs = pl_module.model(batch)

        canonical_true, canonical_generated = generate_canonical_smiles(
            model=pl_module.model,
            tokenizer=tokenizer,
            structure_features=outputs["structure_features"],
            structure_features_mask=outputs["structure_features_mask"],
            true_smiles=batch["molecules"],
            max_length=max_molecule_len,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            num_sequences=NUM_SEQUENCES,
        )
        generations["all_canon_true"].extend(canonical_true)
        generations["all_canon_generated"].extend(canonical_generated)

        if len(generations["all_canon_true"]) >= MAX_MOLECULES:
            break

    results = {
        "generations": generations,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "num_sequences": NUM_SEQUENCES,
        "seed": args.seed,
    }
    output_file = Path(args.model_path) / f"test_results-temperature-{TEMPERATURE}-top_k-{TOP_K}_seed-{args.seed}.json"
    with output_file.open("w") as file:
        json.dump(results, file, indent=4)
    print(f"Saved {len(generations['all_canon_true'])} results to {output_file}")

if __name__ == "__main__":
    main()
