import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
from scipy.signal import find_peaks
import h5py
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from torch_geometric.data import Data as PyGData
from ogb.utils.features import atom_to_feature_vector
from ogb.utils.features import bond_to_feature_vector
from nmiracle_v2.data.alberts_baseline_tokenize import build_source_string as build_alberts_baseline_source_string
from nmiracle_v2.data.mmspectrum_vocab import MMSpectrumVocab

# For formula prior-embeddings to stay meaningful (element -> index mapping)
ATOM_NUM_TO_IDX = {  # no hydrogen
    0: 0,   # Padding
    6: 1,   # Carbon
    7: 2,   # Nitrogen
    8: 3,   # Oxygen
    9: 4,   # Fluorine
    15: 5,  # Phosphorus
    16: 6,  # Sulfur
    17: 7,  # Chlorine
    35: 8,  # Bromine
    53: 9,  # Iodine
}

def process_formula(smiles, max_elements_len):
    """SMILES -> (padded element indices, padded element counts), using
    ATOM_NUM_TO_IDX. For formula prior-embeddings to stay meaningful (element -> index mapping)
    when fine-tuning a spec2struct model with use_formula=True.
    """
    mol = Chem.MolFromSmiles(smiles)
    element_counts = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:  # skip H
            idx = ATOM_NUM_TO_IDX.get(atom.GetAtomicNum())
            element_counts[idx] = element_counts.get(idx, 0) + 1

    elements = np.array(list(element_counts.keys()))
    counts = np.array(list(element_counts.values()))
    sorted_indices = np.argsort(elements)
    sorted_elements = elements[sorted_indices]
    sorted_counts = counts[sorted_indices]

    padded_elements = np.zeros(max_elements_len, dtype=np.int64)
    padded_counts = np.zeros(max_elements_len, dtype=np.int64)
    length = min(len(sorted_elements), max_elements_len)
    padded_elements[:length] = sorted_elements[:length]
    padded_counts[:length] = sorted_counts[:length]

    return (
        torch.tensor(padded_elements, dtype=torch.long),
        torch.tensor(padded_counts, dtype=torch.long),
    )


class AlbertsDataset(Dataset):
    """
    Dataset for NMR2Struct-style processing with concatenated spectra
    """
    def __init__(
        self,
        data_dir,
        split='train',
        tokenizer=None,
        n_ir_features=1800,
        n_hnmr_features=10000,
        n_cnmr_features=10000,
        max_molecule_len=128,
        use_ir=False,
        use_hnmr=True,
        use_cnmr=True,
        cnmr_binary=True,
        cnmr_binary_bins=80,
        use_alberts_baseline_tokens=False,# Tokenized-spectrum encoding (unused)
        alberts_baseline_vocab_path=None,
        alberts_baseline_max_len=512,
        annotated_peaks_path=None,
        # Formula prior (independent of spectral representation choice above --
        use_formula=False,
        max_elements_len=7,
    ):
        self.n_ir_features = n_ir_features
        self.n_hnmr_features = n_hnmr_features
        self.n_cnmr_features = n_cnmr_features
        self.use_ir = use_ir
        self.use_hnmr = use_hnmr
        self.use_cnmr = use_cnmr
        self.cnmr_binary = cnmr_binary
        self.cnmr_binary_bins = cnmr_binary_bins
        self.eps = 1e-6  # noise threshold for continuous (non-binary) CNMR, see _process_spectrum
        self.use_formula = use_formula
        self.max_elements_len = max_elements_len
        self.tokenizer = tokenizer
        self.max_molecule_len = max_molecule_len
        self.data_dir = data_dir
        self.split = split

        # Tokenized-spectrum encoding: alberts_baseline/generate_input.py-style
        self.use_alberts_baseline_tokens = use_alberts_baseline_tokens
        self.alberts_baseline_max_len = alberts_baseline_max_len
        self.alberts_baseline_vocab = None
        if self.use_alberts_baseline_tokens:
            if alberts_baseline_vocab_path is None:
                raise ValueError(
                    "alberts_baseline_vocab_path must be set when use_alberts_baseline_tokens=True "
                    "(build it with scripts/build_alberts_baseline_vocab.py first)."
                )
            self.alberts_baseline_vocab = MMSpectrumVocab.load(alberts_baseline_vocab_path)

        #load SMILES
        smiles_path = f"{data_dir}/smiles.npy"
        self.smiles = np.load(smiles_path, allow_pickle=True)
        self.smiles = [smi.decode('utf-8') if isinstance(smi, bytes) else smi for smi in self.smiles]

        # Load splits
        splits_path = f"{data_dir}/split_indices.p"
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        
        # Get indices for this split
        self.indices = splits[split]

        # Store paths to files (we'll open them when needed)
        self.spectra_path = f"{data_dir}/spectra.h5"
        self.spectra_file = None

        self.annotated_peaks_path = annotated_peaks_path
        self.annotated_peaks_file = None
        self.annotated_peaks_matched = None
        if self.annotated_peaks_path is not None:
            with h5py.File(self.annotated_peaks_path, "r") as f:
                self.annotated_peaks_matched = f["matched"][:]
            n_matched_in_split = int(self.annotated_peaks_matched[self.indices].sum())
            print(
                f"Annotated peaks for {split} split: {n_matched_in_split}/{len(self.indices)} "
                f"({n_matched_in_split / len(self.indices):.1%}) matched to real annotations, "
                f"{len(self.indices) - n_matched_in_split} falling back to proxy peak-picking"
            )

        print(f"Loaded {len(self.indices)} samples for {split} split")
    
    def __len__(self):
        return len(self.indices)

    def _open_spectra_file(self):
        """Open HDF5 file lazily"""
        if self.spectra_file is None:
            self.spectra_file = h5py.File(self.spectra_path, 'r')
        return self.spectra_file

    def _open_annotated_peaks_file(self):
        if self.annotated_peaks_file is None:
            self.annotated_peaks_file = h5py.File(self.annotated_peaks_path, 'r')
        return self.annotated_peaks_file


    @staticmethod
    def _find_peak_ppms(trace, ppm_min, ppm_max):
        """Peak-pick a continuous NMR trace into ppm positions."""
        if len(trace) == 10000:  # continuous spectrum (n_{c,h}nmr_features default)
            peaks, _ = find_peaks(trace, height=0.1 * np.max(trace), distance=5)
            ppm_range = np.linspace(ppm_min, ppm_max, len(trace))
            return ppm_range[peaks]
        else:
            # Already a list of peak positions
            return np.array(trace)

    @classmethod
    def _find_cnmr_peak_ppms(cls, cnmr):
        return cls._find_peak_ppms(cnmr, 0, 220)

    def _pad_token_ids(self, token_ids, max_len, pad_id):
        token_ids = token_ids[:max_len]
        token_ids = token_ids + [pad_id] * (max_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def _real_annotated_peaks(self, data_idx):
        """Look up real h_nmr_peaks/c_nmr_peaks/molecular_formula for this
        sample from annotated_peaks.h5, if it was matched by
        scripts/build_annotated_peaks_companion.py. Returns None if no
        companion file is configured or this index wasn't matched.
        """
        if self.annotated_peaks_matched is None or not self.annotated_peaks_matched[data_idx]:
            return None

        def _s(value):
            return value.decode('utf-8') if isinstance(value, bytes) else value

        f = self._open_annotated_peaks_file()
        return {
            "formula": json.loads(_s(f["molecular_formula"][data_idx])),
            "h_nmr_peaks": json.loads(_s(f["h_nmr_peaks"][data_idx])),
            "c_nmr_peaks": json.loads(_s(f["c_nmr_peaks"][data_idx])),
        }

    def _build_alberts_baseline_tokens(self, ir, cnmr, smiles, data_idx):
        """Ablation 1: alberts_baseline/generate_input.py-style tokenization.
        Uses real annotated h_nmr_peaks/c_nmr_peaks/molecular_formula when
        annotated_peaks_path is configured and this sample was matched (see
        _real_annotated_peaks); otherwise falls back to RDKit-derived formula
        + peak-picked CNMR (no HNMR, since multiplicity/J aren't recoverable
        from the continuous trace). Each modality is only included if its own
        use_ir/use_hnmr/use_cnmr flag is set, same as the continuous encoder.
        """
        real = self._real_annotated_peaks(data_idx)
        formula = None
        if self.use_formula:
            if real is not None:
                formula = real["formula"]
            else:
                mol = Chem.MolFromSmiles(smiles)
                formula = rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""

        h_nmr_peaks = None
        if self.use_hnmr and real is not None:
            h_nmr_peaks = real["h_nmr_peaks"]
        # no proxy fallback for HNMR -- multiplicity/J aren't recoverable from
        # the continuous trace, so use_hnmr=True with no real annotation just
        # omits it rather than fabricating a token for it.

        c_nmr_peaks = None
        if self.use_cnmr:
            if real is not None:
                c_nmr_peaks = real["c_nmr_peaks"]
            else:
                peak_ppms = self._find_cnmr_peak_ppms(cnmr)
                c_nmr_peaks = [{"delta (ppm)": ppm} for ppm in peak_ppms]

        source_string = build_alberts_baseline_source_string(
            formula=formula,
            h_nmr_peaks=h_nmr_peaks,
            c_nmr_peaks=c_nmr_peaks,
            ir_spectrum=ir if self.use_ir else None,
        )

        token_ids = self.alberts_baseline_vocab.encode(source_string)
        return self._pad_token_ids(token_ids, self.alberts_baseline_max_len, self.alberts_baseline_vocab.pad_id)

    def _process_spectrum(self, spectrum):
        """Process spectra in NMR2Struct style by concatenating them"""
        # Prepare concatenated spectral input as in NMR2Struct
        spectral_components = []
        
        ir = spectrum[:1800]
        hnmr = spectrum[1800:11800]
        cnmr = spectrum[11800:]

        if self.use_ir:
            if len(ir) > self.n_ir_features:
                ir = ir[:self.n_ir_features]
            else:
                ir = np.pad(ir, (0, self.n_ir_features - len(ir)), 'constant')
            
            # Normalize
            if np.max(np.abs(ir)) > 0:
                ir = ir / np.max(np.abs(ir))
            
            spectral_components.append(ir)
        
        if self.use_hnmr:
            # Truncate or pad
            if len(hnmr) > self.n_hnmr_features:
                hnmr = hnmr[:self.n_hnmr_features]
            else:
                hnmr = np.pad(hnmr, (0, self.n_hnmr_features - len(hnmr)), 'constant')
            
            # Normalize
            if np.max(np.abs(hnmr)) > 0:
                hnmr = hnmr / np.max(np.abs(hnmr))
            
            # Threshold (NMR2Struct style) unused for now.
            # hnmr = np.where(hnmr > self.eps, hnmr, 0)
            spectral_components.append(hnmr)
        
        if self.use_cnmr:            
            if self.cnmr_binary:
                # Use the same approach as NMR2Struct for processing CNMR spectra
                peak_positions = self._find_cnmr_peak_ppms(cnmr)

                # Create binary representation with specified number of bins
                bin_edges = np.linspace(0, 220, self.cnmr_binary_bins + 1)
                binned_cnmr = np.zeros(self.cnmr_binary_bins)
                
                if len(peak_positions) > 0:
                    # Assign peaks to bins
                    bin_indices = np.digitize(peak_positions, bin_edges) - 1
                    bin_indices = np.clip(bin_indices, 0, self.cnmr_binary_bins - 1)  # Ensure valid indices
                    binned_cnmr[bin_indices] = 1
                
                cnmr = binned_cnmr
            else:
                # Continuous representation - standard processing
                if len(cnmr) > self.n_cnmr_features:
                    cnmr = cnmr[:self.n_cnmr_features]
                else:
                    cnmr = np.pad(cnmr, (0, self.n_cnmr_features - len(cnmr)), 'constant')
                
                # Normalize
                if np.max(np.abs(cnmr)) > 0:
                    cnmr = cnmr / np.max(np.abs(cnmr))
                
                # Threshold
                cnmr = np.where(cnmr > self.eps, cnmr, 0)
            
            spectral_components.append(cnmr)
        
        # Concatenate all spectral components
        if spectral_components:
            concatenated_spectra = np.concatenate(spectral_components)
            return torch.tensor(concatenated_spectra, dtype=torch.float32)
        else:
            # Return empty tensor if no spectra
            return torch.tensor([], dtype=torch.float32)
    
    def _process_single_sample(self, idx):
        """Process a single sample (extracted from your current __getitem__)"""
        data_idx = self.indices[idx]

        spectra_file = self._open_spectra_file()
        spectrum = spectra_file['spectra'][data_idx]
        smiles = self.smiles[data_idx]

        # Process the spectrum
        if self.use_alberts_baseline_tokens:
            processed_spectrum = self._build_alberts_baseline_tokens(
                ir=spectrum[:1800], cnmr=spectrum[11800:], smiles=smiles, data_idx=data_idx
            )
        else:
            processed_spectrum = self._process_spectrum(spectrum)

       # Tokenize SMILES (same as your existing code)
        token_ids = self.tokenizer.encode(
            smiles,
            max_length=self.max_molecule_len,
            padding=None,
            truncation=True,
            return_tensors=None
        ).input_ids


        # Create target and decoder input
        target = token_ids.copy()
        target.append(self.tokenizer.eos_token_id)
        
        decoder_input = [self.tokenizer.start_token_id]
        decoder_input.extend(token_ids)

        # Pad sequences
        if len(target) < self.max_molecule_len:
            target.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(target)))
        if len(decoder_input) < self.max_molecule_len:
            decoder_input.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(decoder_input)))

        result = {
            'molecules': smiles,
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
        }

        result['spectra'] = processed_spectrum
        # Only compute the separate elements/element_counts view when formula
        # isn't already baked into the token string (use_alberts_baseline_tokens)
        # -- avoids representing formula twice.
        if self.use_formula and not self.use_alberts_baseline_tokens:
            elements, element_counts = process_formula(smiles, self.max_elements_len)
            result['elements'] = elements
            result['element_counts'] = element_counts

        return result
    
    def __getitem__(self, idx):
        return self._process_single_sample(idx)


class XueDataset(AlbertsDataset):
    """Experimental IR + 1H spectra measured in-house (Xue et al., FR4-01 plate series).
    Built by scripts/build_xue_h5.py into the SAME (N, 21800) layout as Alberts
    (IR[0:1800] ascending 400-4000 cm^-1, 1H[1800:11800] descending 10 to -2 ppm,
    13C[11800:] zeros), so all of AlbertsDataset's loading and spectrum processing
    applies unchanged and this subclass only records provenance and guards the
    differences that matter:

      * NO 13C.  The last 10000 columns are zeros, not data. use_cnmr=True would
        silently feed an all-zero modality as if it were a measurement, so it raises.
      * IR is measured only down to ~650 cm^-1; the 126 bins below that are zero-filled
        (93% coverage), recorded per row in metadata.csv.
      * Real spectra carry a CDCl3 residual at 7.26 ppm that simulated Alberts data never
        has. build_xue_h5.py --drop_solvent produces a variant with it removed.
    Kept as its own class rather than a data_dir swap so that run names, the datamodule
    dispatch and any future xue-specific handling stay distinct from simulated Alberts.
    """

    def __init__(self, *args, use_cnmr=False, **kwargs):
        if use_cnmr:
            raise ValueError(
                "XueDataset has no 13C: those columns are zeros. Set use_cnmr=False "
                "(passing True would feed an all-zero modality as if it were measured)."
            )
        super().__init__(*args, use_cnmr=False, **kwargs)

class NistIRDataset(Dataset):
    """
    Dataset for experimental NIST IR spectra.

    Expected files in data_dir:
      - nist_spectra.h5 or nist_spectra_edge_fill.h5 with dataset key "spectra"
      - nist_smiles.npy
      - split_indices.p with keys "train", "val", "test"

    Continuous CNN representation by default. Set use_alberts_baseline_tokens=True
    to instead tokenize the raw IR trace with alberts_baseline_tokenize.process_ir
    -- the same function and vocab used for Ablation 1 on simulated Alberts data --
    so a model trained on tokenized simulated IR can be evaluated zero-shot on
    real experimental IR. No HNMR/CNMR exist here, only IR (+ optional formula).
    """
    def __init__(
        self,
        data_dir,
        split="train",
        tokenizer=None,
        n_ir_features=1800,
        max_molecule_len=128,
        use_alberts_baseline_tokens=False,
        alberts_baseline_vocab_path=None,
        alberts_baseline_max_len=512,
        use_formula=False,
    ):

        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.n_ir_features = n_ir_features
        self.max_molecule_len = max_molecule_len

        self.use_alberts_baseline_tokens = use_alberts_baseline_tokens
        self.alberts_baseline_max_len = alberts_baseline_max_len
        self.use_formula = use_formula
        self.alberts_baseline_vocab = None
        if self.use_alberts_baseline_tokens:
            if alberts_baseline_vocab_path is None:
                raise ValueError(
                    "alberts_baseline_vocab_path must be set when use_alberts_baseline_tokens=True "
                    "(reuse the vocab built on Alberts training data for a real generalization test)."
                )
            self.alberts_baseline_vocab = MMSpectrumVocab.load(alberts_baseline_vocab_path)

        smiles_path = self.data_dir / "nist_smiles.npy"
        self.smiles = np.load(smiles_path, allow_pickle=True)
        self.smiles = [
            smi.decode("utf-8") if isinstance(smi, bytes) else str(smi)
            for smi in self.smiles
        ]

        splits_path = self.data_dir / "split_indices.p"
        splits = np.load(splits_path, allow_pickle=True).item()

        self.indices      = np.array(splits[split], dtype=int)
        self.spectra_path = self.data_dir / "nist_spectra.h5"
        self.spectra_file = None

        print(f"Loaded {len(self.indices)} experimental IR samples for {split} split")

    def __len__(self):
        return len(self.indices)

    def _open_spectra_file(self):
        if self.spectra_file is None:
            self.spectra_file = h5py.File(self.spectra_path, "r")
        return self.spectra_file

    def _build_alberts_baseline_ir_tokens(self, ir, smiles):
        """Same tokenization (alberts_baseline_tokenize.process_ir) and vocab
        used for Ablation 1 on simulated Alberts data -- only the IR portion
        (no HNMR/CNMR here), plus formula if use_formula is set.
        """
        formula = None
        if self.use_formula:
            mol = Chem.MolFromSmiles(smiles)
            formula = rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""

        source_string = build_alberts_baseline_source_string(formula=formula, ir_spectrum=ir)
        token_ids = self.alberts_baseline_vocab.encode(source_string)
        token_ids = token_ids[: self.alberts_baseline_max_len]
        token_ids = token_ids + [self.alberts_baseline_vocab.pad_id] * (
            self.alberts_baseline_max_len - len(token_ids)
        )
        return torch.tensor(token_ids, dtype=torch.long)

    def _process_ir(self, ir):
        if len(ir) > self.n_ir_features:
            ir = ir[:self.n_ir_features]
        else:
            ir = np.pad(ir, (0, self.n_ir_features - len(ir)), "constant")

        if np.max(np.abs(ir))> 0:
            ir = ir / np.max(np.abs(ir))

        return torch.tensor(ir, dtype=torch.float32)

    def _tokenize_smiles(self, smiles):
        token_ids = self.tokenizer.encode(
            smiles,
            max_length=self.max_molecule_len,
            padding=None,
            truncation=True,
            return_tensors=None,
        ).input_ids

        target = token_ids.copy()
        target.append(self.tokenizer.eos_token_id)

        decoder_input = [self.tokenizer.start_token_id]
        decoder_input.extend(token_ids)

        if len(target) < self.max_molecule_len:
            target.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(target)))
        else:
            target = target[:self.max_molecule_len]

        if len(decoder_input) < self.max_molecule_len:
            decoder_input.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(decoder_input)))
        else:
            decoder_input = decoder_input[:self.max_molecule_len]

        return (
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )

    def __getitem__(self, idx):
        data_idx = int(self.indices[idx])

        spectra_file = self._open_spectra_file()
        ir = spectra_file["spectra"][data_idx]
        smiles = self.smiles[data_idx]

        decoder_input, target = self._tokenize_smiles(smiles)

        if self.use_alberts_baseline_tokens:
            processed_spectrum = self._build_alberts_baseline_ir_tokens(ir, smiles)
        else:
            processed_spectrum = self._process_ir(ir)

        result = {
            "spectra": processed_spectrum,
            "molecules": smiles,
            "decoder_input": decoder_input,
            "target": target,
            "h5_idx": torch.tensor(data_idx, dtype=torch.long),
        }
        # Substructure counts are consumed only by Sub2Structure (Stage 1); NIST is used
        # exclusively in the spec2struct path, where nothing reads them.
        return result


class NMRSIDataset(Dataset):
    """
    Real experimental NMR (~1.4M molecule instances with both
    1H and 13C reported), prepared by scripts/prepare_nmrsi_dataset.py.
    Only feeds the tokenized-encoder ablation (alberts_baseline_tokenize.py) --
    there's no continuous trace here (literature-reported peak lists only, no
    raw FID) and no IR/HSQC/MS data, so this dataset can't be used with the
    continuous CNN. `vocab_path` is meant to point
    at the vocab already trained on the (simulated) Alberts dataset, so this
    serves as a genuine sim-to-real generalization test, not a fresh in-domain
    fit -- pass a fresh vocab only if you deliberately want to fine-tune on
    NMRSI's own token distribution instead.
    Expected files in data_dir (see prepare_nmrsi_dataset.py):
      - nmrsi_smiles.npy
      - nmrsi_peaks.h5 with datasets "molecular_formula", "h_nmr_peaks", "c_nmr_peaks"
      - split_indices.p with keys "train", "val", "test"
    """
    def __init__(
        self,
        data_dir,
        vocab_path,
        split="train",
        tokenizer=None,
        max_molecule_len=128,
        max_len=512,
        use_hnmr=True,
        use_cnmr=True,
        use_formula=True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_molecule_len = max_molecule_len
        self.max_len = max_len
        self.use_hnmr = use_hnmr
        self.use_cnmr = use_cnmr
        self.use_formula = use_formula
        self.vocab = MMSpectrumVocab.load(vocab_path)

        smiles_path = self.data_dir / "nmrsi_smiles.npy"
        self.smiles = np.load(smiles_path, allow_pickle=True)
        self.smiles = [
            smi.decode("utf-8") if isinstance(smi, bytes) else str(smi)
            for smi in self.smiles
        ]

        splits_path = self.data_dir / "split_indices.p"
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        self.indices = np.array(splits[split], dtype=int)

        self.peaks_path = self.data_dir / "nmrsi_peaks.h5"
        self.peaks_file = None

        print(f"Loaded {len(self.indices)} real experimental NMRSI samples for {split} split")

    def __len__(self):
        return len(self.indices)

    def _open_peaks_file(self):
        if self.peaks_file is None:
            self.peaks_file = h5py.File(self.peaks_path, "r")
        return self.peaks_file

    def _build_tokens(self, data_idx):
        def _s(value):
            return value.decode("utf-8") if isinstance(value, bytes) else value

        f = self._open_peaks_file()
        # Same gating as AlbertsDataset._build_alberts_baseline_tokens: build_source_string
        # concatenates only the parts it is given, so a disabled modality is left out of the
        # string entirely rather than tokenized as empty.
        formula = _s(f["molecular_formula"][data_idx]) if self.use_formula else None
        h_nmr_peaks = json.loads(_s(f["h_nmr_peaks"][data_idx])) if self.use_hnmr else None
        c_nmr_peaks = json.loads(_s(f["c_nmr_peaks"][data_idx])) if self.use_cnmr else None

        source_string = build_alberts_baseline_source_string(
            formula=formula, h_nmr_peaks=h_nmr_peaks, c_nmr_peaks=c_nmr_peaks,
        )
        token_ids = self.vocab.encode(source_string)
        return self._pad_token_ids(token_ids, self.max_len, self.vocab.pad_id)

    def _pad_token_ids(self, token_ids, max_len, pad_id):
        token_ids = token_ids[:max_len]
        token_ids = token_ids + [pad_id] * (max_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def _tokenize_smiles(self, smiles):
        token_ids = self.tokenizer.encode(
            smiles, max_length=self.max_molecule_len, padding=None,
            truncation=True, return_tensors=None,
        ).input_ids

        target = token_ids.copy()
        target.append(self.tokenizer.eos_token_id)
        decoder_input = [self.tokenizer.start_token_id] + token_ids

        if len(target) < self.max_molecule_len:
            target.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(target)))
        else:
            target = target[:self.max_molecule_len]

        if len(decoder_input) < self.max_molecule_len:
            decoder_input.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(decoder_input)))
        else:
            decoder_input = decoder_input[:self.max_molecule_len]

        return (
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )

    def __getitem__(self, idx):
        data_idx = int(self.indices[idx])
        smiles = self.smiles[data_idx]
        decoder_input, target = self._tokenize_smiles(smiles)

        return {
            "spectra": self._build_tokens(data_idx),
            "molecules": smiles,
            "decoder_input": decoder_input,
            "target": target,
        }


class PreTrainDataset(Dataset):
    """
    Unified dataset that supports both lazy loading and RAM preprocessing
    Handles both graph-based and traditional substructure approaches
    """
    
    def __init__(
        self, 
        data_dir,
        indices,
        tokenizer,
        max_molecule_len=None,
        max_substructures_len=None,
        max_substructures_count=None,
        max_elements_len=None,
        max_elements_count=None,
        use_formula=False,
        use_graph=False,
        use_substructures=False,
        substructure_featurizer="vocab",
        morgan_radius=2,
        morgan_n_bits=8192, #in pushing the limits they are using 8192..
        morgan_use_chirality=False,
    ):
        self.data_dir = data_dir
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_molecule_len = max_molecule_len
        self.max_substructures_count  = max_substructures_count
        self.max_substructures_len = max_substructures_len

        self.use_formula = use_formula
        self.use_graph = use_graph
        self.use_substructures = use_substructures
        self.substructure_featurizer = substructure_featurizer
        self.morgan_radius = morgan_radius
        self.morgan_n_bits = morgan_n_bits
        self.morgan_use_chirality = morgan_use_chirality

        if self.substructure_featurizer not in {"vocab", "morgan_counts"}:
            raise ValueError(
                "substructure_featurizer must be either 'vocab' or 'morgan_counts'"
            )

        self.max_elements_count = max_elements_count
        self.max_elements_len = max_elements_len

        # Set up file paths
        self.smiles_path = Path(data_dir) / 'combined_pretrain_smiles.npy'
        self.substructures_path = Path(data_dir) / 'combined_pretrain_substructure_counts.h5'

        # Load SMILES once (always small)
        smiles = np.load(self.smiles_path, allow_pickle=True)[:]
        self.smiles = [smi.decode('utf-8') for smi in smiles]
        
        self.substructures_file = None

    def _open_substructures_file(self):
        """Lazy open substructures file"""
        if self.substructures_file is None:
            self.substructures_file = h5py.File(self.substructures_path, 'r')
        return self.substructures_file


    def _get_substructure_counts(self, idx, sample_idx):
        substructures_file = self._open_substructures_file()
        return substructures_file['counts'][sample_idx]
    
    def _process_smiles_tokens(self, smiles):
        """Process SMILES into tokens (shared logic)"""
        token_ids = self.tokenizer.encode(
            smiles,
            max_length=self.max_molecule_len,
            padding=None,
            truncation=True,
            return_tensors=None
        ).input_ids
        
        # Create target and decoder input
        target = token_ids.copy()
        target.append(self.tokenizer.eos_token_id)
        
        decoder_input = [self.tokenizer.start_token_id]
        decoder_input.extend(token_ids)
        
        # Pad sequences
        if len(target) < self.max_molecule_len:
            target.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(target)))
        if len(decoder_input) < self.max_molecule_len:
            decoder_input.extend([self.tokenizer.pad_token_id] * (self.max_molecule_len - len(decoder_input)))
        
        return {
            'target': torch.tensor(target, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'attention_mask': (torch.tensor(target, dtype=torch.long) != self.tokenizer.pad_token_id).float()
        }
    
    def _smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(atom_to_feature_vector(atom))

        x = torch.tensor(atom_features, dtype=torch.long)

        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [edge_feature, edge_feature]

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

        return PyGData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

    def _process_single_sample(self, idx):
        """Process a single sample on-demand"""
        sample_idx = self.indices[idx]
        smiles = self.smiles[sample_idx]

        # Process SMILES tokens
        token_data = self._process_smiles_tokens(smiles)

        if self.use_substructures:
            if self.substructure_featurizer == "morgan_counts":
                raw_counts = self._morgan_count_fingerprint(smiles)
            else:  # "vocab"
                raw_counts = self._get_substructure_counts(idx, sample_idx)
            substructures, substructure_counts = self._process_substructures_and_counts(raw_counts)

        if self.use_formula:
            elements, element_counts = self._process_formula(smiles)            

        # Base result
        result = {
            'molecules': smiles,
            'decoder_input': token_data['decoder_input'],
            'attention_mask': token_data['attention_mask'],
            'target': token_data['target']
        }

        if self.use_substructures:
            result['substructures'] = substructures
            result['substructure_counts'] = substructure_counts

        if self.use_graph:
            result['graph'] = self._smiles_to_graph(smiles)

        if self.use_formula:
            result['elements'] = elements
            result['element_counts'] = element_counts
            
        return result

    def _process_substructures_and_counts(self, counts):
        nonzero_mask = counts > 0 
        nonzero_indices = np.arange(len(counts))[nonzero_mask] + 1
        nonzero_counts = counts[nonzero_mask]

        #Sort by index (important for consistent processing)
        sorted_indices = np.argsort(nonzero_indices)
        sorted_substructure_indices = nonzero_indices[sorted_indices]
        sorted_counts = nonzero_counts[sorted_indices]

        # Pad with zeros to max_len
        padded_indices = np.zeros(self.max_substructures_len, dtype=np.int64)
        padded_counts = np.zeros(self.max_substructures_len, dtype=np.int64)

        # Fill in actual values
        length = min(len(sorted_substructure_indices), self.max_substructures_len)
        padded_indices[:length] = sorted_substructure_indices[:length]
        padded_counts[:length] = np.clip(
            sorted_counts[:length],
            0,
            self.max_substructures_count,
        )

        return (
            torch.tensor(padded_indices, dtype=torch.long),
            torch.tensor(padded_counts, dtype=torch.long)
        )

    def _morgan_count_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.morgan_n_bits, dtype=np.int64)

        fingerprint = AllChem.GetHashedMorganFingerprint(
            mol,
            radius=self.morgan_radius,
            nBits=self.morgan_n_bits,
            useChirality=self.morgan_use_chirality,
        )

        counts = np.zeros(self.morgan_n_bits, dtype=np.int64)
        for bit_idx, count in fingerprint.GetNonzeroElements().items():
            counts[int(bit_idx)] = int(count)
        return counts

    def _process_formula(self, smiles):
        return process_formula(smiles, self.max_elements_len)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Process data on-demand
        return self._process_single_sample(idx)
    
