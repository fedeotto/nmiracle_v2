import argparse
import numpy as np
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from nmiracle_v2.configuration import build_config
from nmiracle_v2.data.datamodule import SpectralDataModule
from nmiracle_v2.data.tokenizer import BasicSmilesTokenizer
from nmiracle_v2.utils.utils import build_run_name

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train nmiracle spec2struct/sub2struct models")
    # Pretraining priors
    parser.add_argument("--use_graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_formula", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_substructures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_multitask", action=argparse.BooleanOptionalAction, default=False) #this stays False (no multitask anymore in nmiracle_v2)
    parser.add_argument("--substructure_featurizer", type=str, default='morgan_counts', choices=["vocab", "morgan_counts"])

    # Spectral modalities (spec2struct stage)
    parser.add_argument("--use_ir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_hnmr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cnmr", action=argparse.BooleanOptionalAction, default=False)

    # Tokenized-spectrum encoding (mutually exclusive with use_coupling_graph; see AlbertsDatasetConfig)
    parser.add_argument("--use_alberts_baseline_tokens", action=argparse.BooleanOptionalAction, default=False) #unused
    parser.add_argument("--alberts_baseline_vocab_path", type=str, default=None) #unused
    parser.add_argument("--annotated_peaks_path", type=str, default=None) #unused
    parser.add_argument("--dataset", type=str, default='alberts_dataset', choices=["alberts_dataset", "nist_ir_dataset", "nmrsi_dataset", "xue_dataset", "pretrain_dataset"],
                        help="Dataset config class; brings its own data_dir.")
    
    parser.add_argument("--tokenized_spectrum_encoder_vocab_size", type=int, default=None)
    # Pretrained checkpoints
    parser.add_argument("--pretrained_structure_model_path", type=str, default=None) #Checkpoint to the structural model (sub2struct stage)
    parser.add_argument("--pretrained_ir_sim_model_path", type=str, default=None) #Checkpoint to the IR-sim pre-trained model (spec2struct) (for fine-tuning purposes on exp IR)

    # Trainer / optimization
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.0) #no label smoothing in the original nmiracle training. 0.1 used in 
    parser.add_argument("--use_noam_schedule", action=argparse.BooleanOptionalAction, default=False) #When enabled, automatically matches training settings used in
    parser.add_argument("--noam_factor", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None) #lr decided in the configs.
    # parser.add_argument("--batch_size", type=int, default=128) #batch size is decided in the configs (sub2struct vs spec2struct)

    # Logging / checkpointing
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--wandb_enabled", action=argparse.BooleanOptionalAction, default=None)
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_config(args)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    #Setting up the tokenizer
    tokenizer = BasicSmilesTokenizer()
    alphabet  = np.load(config.data.alphabet, allow_pickle=True)
    tokenizer.setup_alphabet(alphabet)

    # Initialize data module
    data_module = SpectralDataModule(config=config, tokenizer=tokenizer)
    filename    = build_run_name(config)

    # Initialize model
    if config.training_stage == 'sub2struct':
        from nmiracle_v3.models.pl_module import Sub2StructModule
        model = Sub2StructModule(config, tokenizer)
    elif config.training_stage == 'spec2struct':
        from nmiracle_v3.models.pl_module import Spec2StructModule
        if config.model.pretrained_structure_model_path is None:
            #Quick warning, no pre-trained stuctural model is being used!
            print("⚠️ WARNING: spec2struct with NO pretrained weights (neither a sub2struct "
                  "prior nor an IR-sim checkpoint) -- training from random initialisation.")
        if config.model.pretrained_ir_sim_model_path is not None:
            #Here we check whether we are providing a pre-trained spectra-to-structure model trained on simulated IR
            #Because we have available an experimental IR dataset derived from NIST.
            print(f"Loading pretrained IR simulated model from {config.model.pretrained_ir_sim_model_path}...")
            ir_sim_state_dict = torch.load(config.model.pretrained_ir_sim_model_path,map_location='cpu', weights_only=False)['state_dict']
            model.load_state_dict(ir_sim_state_dict, strict=True)
            print('Loaded pretrained IR simulated model for fine-tuning! ✅')
            filename += f'-pretrained_ir_sim'
            # Skip the sub2struct prior ONLY when the full spec2struct weights are about to be
            # loaded from an IR-sim checkpoint below
            config.model.pretrained_structure_model_path = None

        model = Spec2StructModule(config, tokenizer)

    loggers  = []
    # Add wandb logger if enabled
    if config.logger.enabled:
        wandb_logger = pl.loggers.WandbLogger(
            name=filename,
            project=config.logger.project,
            entity=config.logger.entity,
            log_model=True)
        loggers.append(wandb_logger)

    # Set up callbacks
    callbacks = []

    resume_from_checkpoint = None
    if hasattr(config.trainer, 'ckpt_path'):
        if config.trainer.ckpt_path is not None and os.path.exists(config.trainer.ckpt_path):
            resume_from_checkpoint = config.trainer.ckpt_path
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Warning: Checkpoint path {config.trainer.ckpt_path} not found, starting fresh training.")

    # When resuming, keep saving into the same directory as the checkpoint we resumed from;
    # otherwise use the freshly computed run name.
    if resume_from_checkpoint is not None and args.pretrained_ir_sim_model_path is None:
        ckpt_dirpath = os.path.dirname(resume_from_checkpoint)
    else:
        ckpt_dirpath = f"important_ckpts/{filename}/"
    print(f"Saving checkpoints to: {ckpt_dirpath}")

    ckpt_filename = "{epoch}-{val_loss:.2f}"

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dirpath,
        filename=ckpt_filename,
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True)

    callbacks.append(checkpoint_callback)

    # Device / distributed setup
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices_per_node = torch.cuda.device_count()
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        accelerator = "cpu"
        devices_per_node = 1
        strategy='auto'

    trainer = pl.Trainer(
        max_steps=config.trainer.max_steps if config.trainer.max_steps is not None else -1,
        max_epochs=config.trainer.max_epochs if config.trainer.max_epochs is not None else None,
        accelerator=accelerator,
        devices=devices_per_node,
        num_nodes = config.trainer.num_nodes,
        deterministic=True,
        precision="16-mixed",
        log_every_n_steps=config.trainer.log_every_n_steps,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        strategy=strategy,
        logger=loggers,
        inference_mode=False,
        callbacks=callbacks)

    # Train model
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)

if __name__ == "__main__":
    main()
