import torch
from nmiracle_v2.utils.eval_utils import canonicalize_smiles

def move_to_device(value, device):
    """Recursively move tensors and graph batches to a device."""
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if hasattr(value, "to") and not isinstance(value, str):
        return value.to(device)
    return value


@torch.no_grad()
def generate_canonical_smiles(
    model,
    tokenizer,
    structure_features,
    structure_features_mask,
    true_smiles,
    max_length,
    temperature,
    top_k,
    num_sequences,
):
    """Generate, score-sort, decode, and canonicalize SMILES for one batch."""
    generated_ids, scores = model.generate(
        structure_features=structure_features,
        structure_features_mask=structure_features_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        num_sequences=num_sequences,
    )

    canonical_generated = []
    for sample_ids, sample_scores in zip(generated_ids, scores):
        sample_smiles = []
        for index in torch.argsort(sample_scores):
            smiles = tokenizer.decode(sample_ids[index])
            canonical_smiles = canonicalize_smiles(smiles)
            if canonical_smiles is not None:
                sample_smiles.append(canonical_smiles)
        canonical_generated.append(sample_smiles)

    canonical_true = [canonicalize_smiles(str(smiles)) for smiles in true_smiles]
    return canonical_true, canonical_generated
