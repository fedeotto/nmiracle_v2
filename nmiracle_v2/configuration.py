from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

def _restore_dataclass(config_class, values):
    """Build a dataclass from checkpoint values while retaining new defaults."""
    if isinstance(values, config_class):
        return values
    if not isinstance(values, Mapping):
        raise TypeError(
            f"Expected checkpoint values for {config_class.__name__} to be a mapping, "
            f"got {type(values).__name__}."
        )

    field_names = {config_field.name for config_field in fields(config_class)}
    return config_class(**{
        key: value for key, value in values.items() if key in field_names
    })

@dataclass
class TrainerConfig:
    max_epochs: int = 300
    max_steps: int = None 
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 50
    ckpt_path: str | None = None

@dataclass
class AlbertsDatasetConfig:
    name: str = "alberts_dataset"
    data_dir: Path = Path("data/alberts_h5")

    max_molecule_len: int = 77
    num_substructures: int = 991
    max_substructures_len: int = 103
    max_substructures_count: int = 232
    max_count_value: int = 232

    n_ir_features: int = 1800
    n_hnmr_features: int = 10000
    n_cnmr_features: int = 10000
    cnmr_binary_bins: int = 80

    use_ir: bool = True
    use_hnmr: bool = True
    use_cnmr: bool = False
    cnmr_binary: bool = True

    use_alberts_baseline_tokens: bool = False
    alberts_baseline_vocab_path: Path | None = None
    alberts_baseline_max_len: int = 512
    annotated_peaks_path: Path | None = None

    use_formula: bool = False
    max_elements_len: int = 7

    bs_train: int = 64
    bs_val: int = 64
    bs_test: int = 64

@dataclass
class NISTIRDatasetConfig:
    name: str = "nist_ir_dataset"
    data_dir: Path = Path("data/nist_h5")
    max_molecule_len: int = 240 #is 77 for Alberts, much longer here o_O
    num_substructures: int = 991
    max_substructures_len: int = 114
    max_substructures_count: int = 232 # 132 just for NIST but we need to fine-tune from Alberts.
    max_count_value: int = 232 #132 just for NIST but we need to fine-tune from Alberts.
    use_ir: bool = True
    n_ir_features: int = 1800
    use_alberts_baseline_tokens: bool = False # Tokenizes IR with the same alberts_baseline_tokenize.process_ir (instead of using continuous CNN)
    alberts_baseline_vocab_path: Path | None = None
    alberts_baseline_max_len: int = 512
    use_formula: bool = False

    bs_train: int= 64
    bs_val: int = 64
    bs_test: int =64

@dataclass
class XueDatasetConfig:
    # In-house experimental IR + 1H
    name: str = "xue_dataset"
    data_dir: Path = Path("data/xue_h5")

    n_ir_features: int = 1800
    n_hnmr_features: int = 10000
    max_molecule_len: int = 128

    use_ir: bool = True
    use_hnmr: bool = True
    #no cnmr in this dataset (?)
    use_alberts_baseline_tokens: bool = False
    alberts_baseline_vocab_path: Path | None = None
    alberts_baseline_max_len: int = 512
    annotated_peaks_path: Path | None = None

    use_formula: bool = False
    max_elements_len: int = 7

    bs_train: int = 64
    bs_val: int = 64
    bs_test: int = 64


@dataclass
class NMRSIDatasetConfig:
    name: str = "nmrsi_dataset"
    data_dir: Path = Path("data/nmrsi_h5")
    max_molecule_len: int = 240
    #This dataset (similar to NIST) is meant for fine-tuning (or zero-shot evaluation) purposes.
    #So we load the vocab used for pre-training on simulated data (Alberts)
    alberts_baseline_vocab_path: Path | None = 'data/alberts_native_h5/alberts_baseline_vocab.json'  # point at the Alberts-trained vocab for a real generalization test
    alberts_baseline_max_len: int = 512

    use_hnmr: bool = True
    use_cnmr: bool = True
    use_formula: bool = True

    bs_train: int = 64
    bs_val: int = 64
    bs_test: int = 64

@dataclass
class PretrainDatasetConfig:
    name: str = "pretrain_dataset"
    data_dir: Path = Path("data/pretrain")

    max_molecule_len: int = 76
    num_substructures: int = 991 #8192 for Morgan.
    max_substructures_len: int = 103 #82 for Morgan 103 for vocab
    max_substructures_count: int = 232 #32 for Morgan 232 for vocab

    # "vocab" or "morgan_counts". The default is only ever reached through from_dict(), for
    # checkpoints saved before this field existed -- and those necessarily used the SMARTS
    # vocabulary, since Morgan was added with the field itself. Defaulting to morgan_counts
    # rebuilt them with an 8192-row embedding against 991 saved rows.
    substructure_featurizer: str = "vocab"
    morgan_radius: int = 2
    morgan_n_bits: int = 8192
    morgan_use_chirality: bool = False

    num_elements: int = 9
    max_elements_len: int = 7
    max_elements_count: int = 35
    alphabet: Path | None = None

    bs_train: int = 1024
    bs_val: int = 64
    bs_test: int = 64

@dataclass
class NMIRacleDataConfig:
    training_stage: str | None = "spec2struct"
    dataset: AlbertsDatasetConfig | PretrainDatasetConfig | NISTIRDatasetConfig | NMRSIDatasetConfig = field(default_factory=AlbertsDatasetConfig)
    alphabet: Path | None = 'important_ckpts/pretrain_mynmr2struct_new/alphabet.npy'
    num_workers: int = 8
    prefetch_factor: int | None = None
    
@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    layer_norm_eps: float = 1e-5
    batch_first: bool = True
    norm_first: bool = False
    dropout: float = 0.1

@dataclass
class MultiSpectraEncoderConfig:
    d_model: int = 128
    nhead: int = 4
    fusion_layers: int = 2
    spectrum_encoder_layers: int = 2
    pool_size_1: int = 12
    out_channels_1: int = 64
    kernel_size_1: int = 5
    pool_size_2: int = 20
    out_channels_2: int = 128
    kernel_size_2: int = 9
    add_pos_encode: bool = True
    fusion_scheme: str = "transformer"
    shared_encoder: bool = False
    n_ir_features: int = 0 #1800
    n_hnmr_features: int = 0 #10000
    n_cnmr_features: int = 0 #10000
    use_ir: bool = False #activated by data + representation chosen
    use_hnmr: bool = False #activated by data + representation chosen
    use_cnmr: bool = False #activated by data + representation chosen
    cnmr_binary: bool = False #activated by data + representation chosen
    cnmr_binary_bins: int = 0 #80 activated by data + representation chosen

@dataclass
class TokenizedSpectrumEncoderConfig:
    # vocab_size must match the vocab built by scripts/build_alberts_baseline_vocab.py
    vocab_size: int = 6000 #32768
    pad_id: int = 1  # MMSpectrumVocab always places <blank> at index 1
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    add_pos_encode: bool = True
    max_len: int = 512

@dataclass
class OptimizerConfig:
    """
    Noam warmup+decay LR schedule (the recipe alberts_baseline/MM-Spectrum's
    OpenNMT training actually uses: linear warmup then inverse-sqrt decay,
    scaled by d_model). Off by default to keep existing flat-LR runs
    unchanged; when enabled, `noam_factor` replaces `lr` as the optimizer's
    base LR (OpenNMT's own recipe uses noam_factor=2.0, warmup_steps=8000 --
    very different scale from a flat Adam lr like 1e-5, since the schedule
    multiplies it by d_model^-0.5 and a step-dependent term).
    """
    lr: float = 1e-5
    label_smoothing: float = 0.0 #Original OpenNMT on Alberts uses 0.1
    use_noam_schedule: bool = False
    noam_factor: float = 2.0
    warmup_steps: int = 8000

@dataclass
class GraphEncoderConfig:
    gnn_num_layer: int = 5
    gnn_emb_dim: int = 300
    gnn_JK: str = "last"
    gnn_drop_ratio: float = 0.0
    gnn_type: str = "gin"
    pretrained_path: str | None = ''   # path to GraphMVP checkpoint
    freeze: bool = True

@dataclass
class NMIRacleConfig:
    training_stage: str = "spec2struct"  # or "sub2struct"
    structure_model: TransformerConfig = field(default_factory=TransformerConfig)
    pretrained_structure_model_path: str | None = None
    pretrained_spectrum_encoder_path : str | None = None
    pretrained_ir_sim_model_path: str | None = None #'/Users/fottoman/Desktop/imperial/my_multimodal/important_ckpts/nmiracle-spec2struct-alberts_dataset-use_substructuresFalse-sub_feat_none-use_formulaTrue-use_graphTrue-use_irTrue-use_hnmrFalse-use_cnmrFalse-use_MTaskFalse-bs16-lr1e-05/last.ckpt'
    use_multitask: bool = False
    # Exactly one of these survives __post_init__ -- the one whose encoder this run builds.
    # The other is set to None, so the config (and the checkpoint saved from it) never
    # describes a component the model does not have. sub2struct drops both: it has no spectra.
    multispectra_encoder: MultiSpectraEncoderConfig | None = field(default_factory=MultiSpectraEncoderConfig)
    use_tokenized_spectrum_encoder: bool = False
    tokenized_spectrum_encoder: TokenizedSpectrumEncoderConfig | None = field(default_factory=TokenizedSpectrumEncoderConfig)
    use_spectral_graph_encoder: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # Multi-view pre-training options
    use_formula: bool = True
    use_graph: bool = False
    use_qformer: bool = False
    use_substructures: bool = False
    graph_encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)

@dataclass
class LoggerConfig:
  offline: bool = False
  project: str = "nmiracle_v2"
  entity: str = ""
  enabled: bool = False

@dataclass
class ExperimentConfig:
    training_stage: str = "spec2struct"
    model: NMIRacleConfig = field(default_factory=NMIRacleConfig)
    data: NMIRacleDataConfig = field(default_factory=NMIRacleDataConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    seed: int = 42

    @classmethod
    def from_dict(cls, values):
        """Restore the nested dataclass config saved in a Lightning checkpoint."""
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise TypeError(
                f"Expected checkpoint config to be a mapping, got {type(values).__name__}."
            )

        values = dict(values)
        if isinstance(values.get("config"), Mapping):
            values = dict(values["config"])
        training_stage = values.get("training_stage", "spec2struct")

        data_values = dict(values.get("data", {}))
        dataset_values = data_values.get("dataset", {})
        dataset_name = (
            dataset_values.get("name")
            if isinstance(dataset_values, Mapping)
            else getattr(dataset_values, "name", None)
        )
        dataset_classes = {
            "alberts_dataset": AlbertsDatasetConfig,
            "nist_ir_dataset": NISTIRDatasetConfig,
            "nmrsi_dataset": NMRSIDatasetConfig,
            "xue_dataset": XueDatasetConfig,
            "pretrain_dataset": PretrainDatasetConfig,
        }
        dataset_class = dataset_classes.get(
            dataset_name,
            PretrainDatasetConfig if training_stage == "sub2struct" else AlbertsDatasetConfig,
        )
        data_values["dataset"] = _restore_dataclass(dataset_class, dataset_values)
        values["data"] = _restore_dataclass(NMIRacleDataConfig, data_values)

        model_values = dict(values.get("model", {}))
        nested_model_configs = {
            "structure_model": TransformerConfig,
            "multispectra_encoder": MultiSpectraEncoderConfig,
            "tokenized_spectrum_encoder": TokenizedSpectrumEncoderConfig,
            "optimizer": OptimizerConfig,
            "graph_encoder": GraphEncoderConfig,
        }
        for key, config_class in nested_model_configs.items():
            if model_values.get(key) is not None:
                model_values[key] = _restore_dataclass(config_class, model_values[key])
        values["model"] = _restore_dataclass(NMIRacleConfig, model_values)

        for key, config_class in {
            "logger": LoggerConfig,
            "trainer": TrainerConfig,
        }.items():
            if key in values:
                values[key] = _restore_dataclass(config_class, values[key])

        return _restore_dataclass(cls, values)

    def __post_init__(self):
        if self.training_stage not in {"spec2struct", "sub2struct"}:
            raise ValueError(f"Invalid training stage: {self.training_stage}. Must be 'spec2struct' or 'sub2struct'.")
        ds = self.data.dataset

        if hasattr(ds, "use_alberts_baseline_tokens"):
            self.model.use_tokenized_spectrum_encoder = ds.use_alberts_baseline_tokens
        elif ds.name == "nmrsi_dataset":
            # NMRSIDatasetConfig (real experimental NMR) ships peak lists, not continuous
            # traces: the tokenized encoder is the only pathway it supports.
            print("Using NMRSI dataset: enabling tokenized spectrum encoder 🫡 ✅.")
            self.model.use_tokenized_spectrum_encoder = True

        if self.training_stage == "sub2struct":
            # No spectra at all here: Sub2Structure conditions on R(M) and builds neither.
            self.model.multispectra_encoder = None
            self.model.tokenized_spectrum_encoder = None
        elif self.model.use_tokenized_spectrum_encoder:
            self.model.multispectra_encoder = None
            if hasattr(ds, "alberts_baseline_max_len"):
                self.model.tokenized_spectrum_encoder.max_len = ds.alberts_baseline_max_len
        else:
            self.model.tokenized_spectrum_encoder = None
            # The encoder must be built for what the loader will actually produce, so its
            # spectral geometry is copied from the dataset config: the field names are
            # identical on both sides by design, so the shared ones ARE that geometry.
            enc = self.model.multispectra_encoder
            for name in {f.name for f in fields(enc)} & {f.name for f in fields(ds)}:
                setattr(enc, name, getattr(ds, name))

def build_config(args):
    def _kw(**pairs):
        return {k: v for k, v in pairs.items() if v is not None}

    if args.dataset == "pretrain_dataset":
        dataset = PretrainDatasetConfig(**_kw(substructure_featurizer=args.substructure_featurizer))
    elif args.dataset == "nmrsi_dataset":
        dataset = NMRSIDatasetConfig(**_kw(alberts_baseline_vocab_path=args.alberts_baseline_vocab_path,
                                           use_hnmr=args.use_hnmr, use_cnmr=args.use_cnmr, use_formula=args.use_formula))
    elif args.dataset == "nist_ir_dataset":
        dataset = NISTIRDatasetConfig(**_kw(alberts_baseline_vocab_path=args.alberts_baseline_vocab_path,
                                            use_ir=args.use_ir, use_formula=args.use_formula))
    elif args.dataset == "alberts_dataset":
        dataset = AlbertsDatasetConfig(**_kw(alberts_baseline_vocab_path=args.alberts_baseline_vocab_path,
                                             use_ir=args.use_ir, use_hnmr=args.use_hnmr, use_cnmr=args.use_cnmr,
                                             use_alberts_baseline_tokens=args.use_alberts_baseline_tokens, annotated_peaks_path=args.annotated_peaks_path,
                                             use_formula=args.use_formula))
    elif args.dataset == "xue_dataset":
        dataset = XueDatasetConfig(**_kw(alberts_baseline_vocab_path=args.alberts_baseline_vocab_path,
                                         use_ir=args.use_ir, use_hnmr=args.use_hnmr,
                                         use_alberts_baseline_tokens=args.use_alberts_baseline_tokens, annotated_peaks_path=args.annotated_peaks_path,
                                         use_formula=args.use_formula))

    return ExperimentConfig(
        training_stage="sub2struct" if args.dataset == "pretrain_dataset" else "spec2struct",
        model=NMIRacleConfig(**_kw(
                use_graph=args.use_graph, use_formula=args.use_formula,
                use_substructures=args.use_substructures, use_multitask=args.use_multitask,
                pretrained_structure_model_path=args.pretrained_structure_model_path,
                pretrained_ir_sim_model_path=args.pretrained_ir_sim_model_path),
            optimizer=OptimizerConfig(**_kw(
                lr=args.lr, label_smoothing=args.label_smoothing,
                use_noam_schedule=args.use_noam_schedule,
                noam_factor=args.noam_factor, warmup_steps=args.warmup_steps)),
            tokenized_spectrum_encoder=TokenizedSpectrumEncoderConfig(**_kw(
                vocab_size=args.tokenized_spectrum_encoder_vocab_size))),
        data=NMIRacleDataConfig(dataset=dataset, **_kw(num_workers=args.num_workers)),
        trainer=TrainerConfig(**_kw(
            max_epochs=args.max_epochs, max_steps=args.max_steps,
            accumulate_grad_batches=args.accumulate_grad_batches,
            num_nodes=args.num_nodes, ckpt_path=args.ckpt_path)),
        logger=LoggerConfig(**_kw(project=args.project, enabled=args.wandb_enabled)))
