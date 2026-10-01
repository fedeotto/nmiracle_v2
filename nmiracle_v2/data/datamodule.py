import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from nmiracle_v2.data.datasets import AlbertsDataset,PreTrainDataset, NistIRDataset, NMRSIDataset, XueDataset
from nmiracle_v2.utils.utils import seed_worker
import numpy as np
from pathlib import Path
import h5py
from torch_geometric.data import Batch as PyGBatch

class SpectralDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.num_workers = config.data.num_workers
        self.prefetch_factor = config.data.prefetch_factor
        self.tokenizer = tokenizer
        self.training_stage = config.training_stage
        
        # Set up data splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.generator = torch.Generator()
        self.generator.manual_seed(0)

    def setup(self, stage=None):
        """Set up datasets for training, validation, and testing"""
        if self.training_stage == "sub2struct":
            print(f"Setting up {self.config.data.dataset.name} dataset for substructure-to-structure training...")
            self._setup_pretrain_dataset(stage=stage)
        
        elif self.training_stage == "spec2struct":
            if self.config.data.dataset.name == 'alberts_dataset':
                print("Setting up Alberts dataset...")
                self._setup_alberts_spec2struct(stage=stage)
            
            elif self.config.data.dataset.name == 'nist_ir_dataset':
                print("Setting up NIST IR dataset...")
                self._setup_nist_ir_spec2struct(stage=stage)

            elif self.config.data.dataset.name == 'xue_dataset':
                print("Setting up Xue experimental IR + 1H dataset...")
                self._setup_xue_spec2struct(stage=stage)

            elif self.config.data.dataset.name == 'nmrsi_dataset':
                print("Setting up NMRSI experimental NMR dataset...")
                self._setup_nmrsi_spec2struct(stage=stage)
            else:
                raise ValueError(f"Unknown dataset_name: {self.config.data.dataset.name}")

    def _setup_pretrain_dataset(self, stage=None):
        # Get paths from config
        data_dir = self.config.data.dataset.data_dir

        # Check for existing indices
        train_indices_path = Path(data_dir) / "sub2struct_train_indices.npy"
        val_indices_path = Path(data_dir) / "sub2struct_val_indices.npy"
        test_indices_path = Path(data_dir) / "sub2struct_test_indices.npy"

        train_indices = np.load(train_indices_path)
        val_indices = np.load(val_indices_path)
        test_indices = np.load(test_indices_path)

        if self.config.data.dataset.max_molecule_len is None:
            smiles = np.load(Path(data_dir) / "combined_pretrain_smiles.npy", allow_pickle=True, mmap_mode="r")
            smiles = [smi.decode('utf-8') if isinstance(smi, bytes) else smi for smi in smiles]
            max_molecule_len = 0
            for i in range(len(smiles)):
                tokens = self.tokenizer.tokenize(smiles[i])
                max_molecule_len = max(max_molecule_len, len(tokens))
            max_molecule_len += 2  # Add space for START and EOS tokens
            del smiles  # Free memory
        else:
            max_molecule_len = self.config.data.dataset.max_molecule_len

        # Calculate max_len for substructures if not provided
        if self.config.data.dataset.max_substructures_len is None:
            substructures_file = h5py.File(Path(data_dir) / "combined_pretrain_substructure_counts.h5", "r")
            substructures      = substructures_file["counts"][:]

            # Calculating max number of substructures per molecule
            counts_per_molecule = np.sum(substructures > 0, axis=1)
            max_count_per_molecule = np.max(counts_per_molecule)
            max_substructures_len = int(max_count_per_molecule * 1.01) # add margin
            print(f"Calculated max_substructures_len: {max_substructures_len}")

            #Calculate max count value
            max_count_value = int(np.max(substructures))
            print(f"Maximum count value for any substructure: {max_count_value}")
        else:
            max_substructures_len = self.config.data.dataset.max_substructures_len
        
        if self.config.data.dataset.max_substructures_count is None:
            substructures_file = h5py.File(Path(data_dir) / "combined_pretrain_substructure_counts.h5", "r")
            substructures      = substructures_file["counts"][:]

            max_count_value = int(np.max(substructures))
            print(f"Calculated max_count_value: {max_count_value}")
            max_count_value = int(max_count_value * 1.01) # add small margin
        else:
            max_count_value = self.config.data.dataset.max_substructures_count

        max_elements_len = self.config.data.dataset.max_elements_len
        max_elements_count = self.config.data.dataset.max_elements_count

        # Common dataset arguments
        common_args = {
            "data_dir": data_dir,
            "tokenizer": self.tokenizer,
            "max_molecule_len": self.config.data.dataset.max_molecule_len, 
            "max_substructures_len": max_substructures_len,
            "max_substructures_count": max_count_value,
            "max_elements_len": max_elements_len,
            "max_elements_count": max_elements_count,
            "use_formula": self.config.model.use_formula,
            "use_graph": self.config.model.use_graph,
            "use_substructures": self.config.model.use_substructures,
            "substructure_featurizer": self.config.data.dataset.substructure_featurizer,
            "morgan_radius": self.config.data.dataset.morgan_radius,
            "morgan_n_bits": self.config.data.dataset.morgan_n_bits,
            "morgan_use_chirality": self.config.data.dataset.morgan_use_chirality,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = PreTrainDataset(indices=train_indices, **common_args)
            self.val_dataset = PreTrainDataset(indices=val_indices, **common_args)

        if stage == "test" or stage is None:
            self.test_dataset = PreTrainDataset(indices=test_indices, **common_args)

    def _setup_alberts_spec2struct(self, stage=None):
        """Set up datasets for spectra-to-all training"""
        ds = self.config.data.dataset
        common_args = {
            'data_dir': ds.data_dir,
            'tokenizer': self.tokenizer,
            'n_ir_features': ds.n_ir_features,
            'n_hnmr_features': ds.n_hnmr_features,
            'n_cnmr_features': ds.n_cnmr_features,
            'max_molecule_len': ds.max_molecule_len,
            'use_ir': ds.use_ir,
            'use_hnmr': ds.use_hnmr,
            'use_cnmr': ds.use_cnmr,
            'cnmr_binary': ds.cnmr_binary,
            'cnmr_binary_bins': ds.cnmr_binary_bins,
            'use_alberts_baseline_tokens': ds.use_alberts_baseline_tokens,
            'alberts_baseline_vocab_path': ds.alberts_baseline_vocab_path,
            'alberts_baseline_max_len': ds.alberts_baseline_max_len,
            'annotated_peaks_path': ds.annotated_peaks_path,
            'use_formula': self.config.model.use_formula,
            'max_elements_len': ds.max_elements_len,
        }

        if stage == 'fit' or stage is None:
            self.train_dataset = AlbertsDataset(split='train', **common_args)
            self.val_dataset = AlbertsDataset(split='val', **common_args)

        if stage == 'test' or stage is None:
            self.test_dataset = AlbertsDataset(split='test', **common_args)

    def _setup_xue_spec2struct(self, stage=None):
        """Xue experimental data IR + 1H"""
        ds = self.config.data.dataset
        common_args = {
            'data_dir': ds.data_dir,
            'tokenizer': self.tokenizer,
            'n_ir_features': ds.n_ir_features,
            'n_hnmr_features': ds.n_hnmr_features,
            'max_molecule_len': ds.max_molecule_len,
            'use_ir': ds.use_ir,
            'use_hnmr': ds.use_hnmr,
            'use_alberts_baseline_tokens': ds.use_alberts_baseline_tokens,
            'alberts_baseline_vocab_path': ds.alberts_baseline_vocab_path,
            'alberts_baseline_max_len': ds.alberts_baseline_max_len,
            'annotated_peaks_path': ds.annotated_peaks_path,
            'use_formula': self.config.model.use_formula,
            'max_elements_len': ds.max_elements_len,
        }

        if stage == 'fit' or stage is None:
            self.train_dataset = XueDataset(split='train', **common_args)
            self.val_dataset = XueDataset(split='val', **common_args)

        if stage == 'test' or stage is None:
            self.test_dataset = XueDataset(split='test', **common_args)


    def _setup_nist_ir_spec2struct(self, stage=None):
        """Set up datasets for spectra-to-all training"""        
        ds = self.config.data.dataset
        common_args = {
            'data_dir': ds.data_dir,
            'tokenizer': self.tokenizer,
            'n_ir_features': ds.n_ir_features,
            'max_molecule_len': ds.max_molecule_len,
            'use_alberts_baseline_tokens': ds.use_alberts_baseline_tokens,
            'alberts_baseline_vocab_path': ds.alberts_baseline_vocab_path,
            'alberts_baseline_max_len': ds.alberts_baseline_max_len,
            'use_formula': self.config.model.use_formula,
        }

        if stage == 'fit' or stage is None:
            self.train_dataset = NistIRDataset(split='train', **common_args)
            self.val_dataset = NistIRDataset(split='val', **common_args)

        if stage == 'test' or stage is None:
            self.test_dataset = NistIRDataset(split='test', **common_args)

    def _setup_nmrsi_spec2struct(self, stage=None):
        """Real experimental NMR -- tokenized-encoder ablation only, see NMRSIDataset."""
        common_args = {
            'data_dir': self.config.data.dataset.data_dir,
            'vocab_path': self.config.data.dataset.alberts_baseline_vocab_path,
            'use_hnmr': self.config.data.dataset.use_hnmr,
            'use_cnmr': self.config.data.dataset.use_cnmr,
            'use_formula': self.config.model.use_formula,
            'tokenizer': self.tokenizer,
            'max_molecule_len': self.config.data.dataset.max_molecule_len,
            'max_len': self.config.data.dataset.alberts_baseline_max_len,
        }

        if stage == 'fit' or stage is None:
            self.train_dataset = NMRSIDataset(split='train', **common_args)
            self.val_dataset = NMRSIDataset(split='val', **common_args)

        if stage == 'test' or stage is None:
            self.test_dataset = NMRSIDataset(split='test', **common_args)

    def _get_collate_fn(self):
        # config.model, not a getattr on the whole config: the flag lives under model, so a
        # defaulting getattr here silently returned False and dropped the graphs.
        return collate_with_graphs if self.config.model.use_graph else None

    @staticmethod
    def _per_split(value, split, default=None):
        if value is not None and hasattr(value, "get") and not isinstance(value, (int, float)):
            return value.get(split, default)
        return value

    def _loader_kwargs(self, split):
        nw = int(self._per_split(self.num_workers, split, 0) or 0)
        kw = dict(num_workers=nw, pin_memory=True)
        if nw > 0:
            kw["persistent_workers"] = True
            pf = self._per_split(self.prefetch_factor, split, None)
            if pf is not None:
                kw["prefetch_factor"] = int(pf)
        return kw

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.dataset.bs_train,
            shuffle=True,
            generator=self.generator,
            worker_init_fn=seed_worker,
            collate_fn=self._get_collate_fn(),
            **self._loader_kwargs("train"),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.dataset.bs_val,
            shuffle=False,
            generator=self.generator,
            worker_init_fn=seed_worker,
            collate_fn=self._get_collate_fn(),
            **self._loader_kwargs("val"),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.dataset.bs_test,
            shuffle=False,
            generator=self.generator,
            worker_init_fn=seed_worker,
            collate_fn=self._get_collate_fn(),
            **self._loader_kwargs("test"),
        )


    # Add helper method for processing substructures
    def _process_substructures(self, substructures, max_len):
        """
        Process substructures according to NMR2Struct's approach:
        Convert binary vector to indices of 1s with 0 padding
        """
        # Convert to numpy array if needed
        if isinstance(substructures, list):
            substructures = np.array(substructures, dtype=np.float32)
        
        # Convert binary vector to indices
        indices = np.arange(len(substructures)) + 1
        indices = indices * substructures
        nonzero_entries = indices[indices != 0]
        
        # Pad with zeros to max_len
        padded = np.zeros(max_len, dtype=np.int64)
        padded[:len(nonzero_entries)] = nonzero_entries
        
        return torch.tensor(padded, dtype=torch.long)


def collate_with_graphs(batch):
    """
    Custom collate that handles batches containing PyG Data objects under the
    'graph'
    """
    from torch.utils.data._utils.collate import default_collate

    pyg_batches = {}
    if 'graph' in batch[0] and batch[0]['graph'] is not None:
        pyg_batches['graph'] = PyGBatch.from_data_list([item['graph'] for item in batch])

    items_without_pyg = [
        {k: v for k, v in item.items() if k not in pyg_batches}
        for item in batch
    ] if pyg_batches else batch

    collated = default_collate(items_without_pyg)
    collated.update(pyg_batches)

    return collated
