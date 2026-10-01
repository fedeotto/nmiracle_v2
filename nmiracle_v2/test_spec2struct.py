import json
import os
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from nmiracle_v2.data.datamodule import SpectralDataModule
from nmiracle_v2.data.tokenizer import BasicSmilesTokenizer
from nmiracle_v2.configuration import ExperimentConfig
from nmiracle_v2.models.pl_module import Spec2StructModule
from nmiracle_v2.utils.generation import generate_canonical_smiles, move_to_device
os.environ["HYDRA_FULL_ERROR"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).parent.parent / "important_ckpts" / "spec2struct-alberts-subvocab-formulasub2struct-ir-hnmr-cnmr-cnn/"
ALPHABET_PATH = 'nmiracle_v2/utils/smiles_alphabet.npy'
CKPT_NAME = "last.ckpt"
ZERO_SHOT_NIST = False
NIST_DATA_DIR = "data/nist_h5"
TEMPERATURE = 1.0
TOP_K = 5
NUM_SEQUENCES = 15
SEED = 0
MAX_MOLECULES = 1000000000 # all test set here.
BATCH_SIZE = 8

def parse_arguments():
    parser = ArgumentParser(description="Test a Spec2Struct model on a dataset.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model directory containing the checkpoint.")
    parser.add_argument("--ckpt_name", type=str, default=CKPT_NAME, help="Name of the checkpoint file to load.")
    parser.add_argument("--alphabet_path", type=str, default=ALPHABET_PATH, help="Path to the alphabet .npy file for the tokenizer.")
    parser.add_argument("--zero_shot_nist", default=ZERO_SHOT_NIST, help="Enable zero-shot testing on NIST IR dataset.")
    parser.add_argument("--nist_data_dir", type=str, default=NIST_DATA_DIR, help="Path to the NIST IR dataset directory.")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()

def load_model_and_checkpoint(model_path, ckpt_name):
    ckpt_path = Path(model_path) / ckpt_name
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "hyper_parameters" not in checkpoint:
        raise KeyError("No hyperparameters found in checkpoint")

    config = ExperimentConfig.from_dict(checkpoint["hyper_parameters"])
    return config, checkpoint

def main():
    args = parse_arguments()
    config, checkpoint = load_model_and_checkpoint(args.model_path, args.ckpt_name)
    pl.seed_everything(args.seed)
    tokenizer = BasicSmilesTokenizer()
    tokenizer.setup_alphabet(np.load(args.alphabet_path, allow_pickle=True))

    if args.zero_shot_nist:
        from nmiracle_v3.configuration import NISTIRDatasetConfig
        src_ds = config.data.dataset
        assert src_ds.use_ir and not (src_ds.use_hnmr or src_ds.use_cnmr), (
            "Zero-shot on NIST requires an IR-ONLY checkpoint: NIST has no 1H/13C spectra, "
            f"but this model was trained with use_ir={src_ds.use_ir}, "
            f"use_hnmr={src_ds.use_hnmr}, use_cnmr={src_ds.use_cnmr}."
        )
        nist_ds = NISTIRDatasetConfig()
        nist_ds.data_dir = args.nist_data_dir
        # These follow the TRAINED MODEL, not the NIST defaults: the decoder was trained
        # with Alberts' sequence length (NIST's default is 240 vs Alberts' 77), and the
        # spectral encoder was built for a fixed set of modalities.
        nist_ds.use_ir, nist_ds.use_hnmr, nist_ds.use_cnmr = (
            src_ds.use_ir, src_ds.use_hnmr, src_ds.use_cnmr)
        config.data.dataset = nist_ds        # NOTE: config.model is deliberately untouched
        print(f"ZERO-SHOT -> NIST ({args.nist_data_dir}), max_molecule_len pinned to "
              f"{nist_ds.max_molecule_len} (the model's training length)")

    config.model.pretrained_structure_model_path = None #final spec2struct model is loaded anyway, we don't need this.
    config.model.use_formula = False # we never use formula in spec2struct in the end only eventually at pre-training.
    config.data.dataset.bs_test = 8
    pl_module = Spec2StructModule(config=config, tokenizer=tokenizer).to(DEVICE)
    pl_module.load_state_dict(checkpoint["state_dict"], strict=True)
    pl_module.eval()
    print("Model state loaded from checkpoint. ✅")

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

    filename = f"test_results-temperature-{TEMPERATURE}-top_k-{TOP_K}_seed-{args.seed}.json"
    if args.zero_shot_nist:
        filename = "nist-zshot-" + filename
    output_file = Path(args.model_path) / filename

    with output_file.open("w") as file:
        json.dump(results, file, indent=4)

    print(f"Saved {len(generations['all_canon_true'])} results to {output_file}")

if __name__ == "__main__":
    main()
