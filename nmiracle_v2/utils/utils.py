import torch
import numpy as np
import random
from pathlib import Path

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _active_tags(pairs):
    return [tag for tag, active in pairs if active]

def _sub_tag(ds):
    """Which fragment featurizer is in use -- the run name must distinguish these, since
    'vocab' (991 SMARTS patterns) and 'morgan_counts' are different representations."""
    return "submorgan" if getattr(ds, "substructure_featurizer", "vocab") == "morgan_counts" else "subvocab"


def build_run_name(config):
    ds = config.data.dataset
    if config.training_stage == "sub2struct":
        priors = _active_tags([
            ("formula", config.model.use_formula),
            ("graph", config.model.use_graph),
            (_sub_tag(ds), config.model.use_substructures),
        ])
        return f"sub2struct-{'-'.join(priors) if priors else 'none'}"

    if config.model.pretrained_structure_model_path is not None:
        pretrained_parts = Path(config.model.pretrained_structure_model_path).parent.name.split("-")
        pretrain_tags = _active_tags([
            ("formulasub2struct", "formula" in pretrained_parts),
            ("graph", "graph" in pretrained_parts),
            ("subvocab", "subvocab" in pretrained_parts),
            ("submorgan", "submorgan" in pretrained_parts),
            # legacy checkpoints predating the featurizer-aware naming
            ("substructures", "substructures" in pretrained_parts),
        ])
    else:
        pretrain_tags = []
    tags = list(pretrain_tags)
    if config.model.use_formula:
        tags.append("formulaspec2struct")

    # Per modality, not per dataset: NMRSIDatasetConfig declares use_hnmr/use_cnmr but no
    # use_ir (no IR data exists for it), and gating the whole block on use_ir would give
    # every NMRSI ablation the same name -- i.e. the same checkpoint directory.
    tags += _active_tags([(m, getattr(ds, f"use_{m}", False)) for m in ("ir", "hnmr", "cnmr")])
    if getattr(ds, "use_alberts_baseline_tokens", False):
        tags.append("spectokens")
    elif hasattr(ds, "use_ir"):
        tags.append("cnn")
    else:
        tags.append("spectokens")  # e.g. nmrsi_dataset: tokenized-only, no continuous pathway

    dataset_short = ds.name.replace("_dataset", "")
    name = f"spec2struct-{dataset_short}-{'-'.join(tags)}"

    return name


            