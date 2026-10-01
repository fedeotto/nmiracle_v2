from dataclasses import asdict
import torch
import pytorch_lightning as pl
from nmiracle_v2.models.sub2structmodel import Sub2Structure
from nmiracle_v2.models.spec2structmodel_v2 import Spectra2Structure
from nmiracle_v2.configuration import ExperimentConfig
import gc
from rdkit import RDLogger
import torch.nn as nn
RDLogger.DisableLog('rdApp.*')

def _noam_lr_lambda(step, d_model, warmup_steps):
    """Attention Is All You Need's warmup+decay schedule -- the recipe
    alberts_baseline/MM-Spectrum's OpenNMT training actually uses
    (decay_method: noam): linear warmup for `warmup_steps`, then inverse-sqrt
    decay, both scaled by d_model^-0.5. `step` is 0-indexed by PyTorch's
    LambdaLR, so it's floored at 1 here to avoid a step^-0.5 blowup at step 0.
    """
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

def _build_optimizer_output(module, optimizer):
    """Wraps `optimizer` with the noam scheduler when
    config.model.optimizer.use_noam_schedule is set; otherwise returns a flat
    LR, unchanged from prior behavior.
    """
    opt_config = module.config.model.optimizer
    if not opt_config.use_noam_schedule:
        return {'optimizer': optimizer}

    d_model = module.config.model.structure_model.d_model
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _noam_lr_lambda(step, d_model, opt_config.warmup_steps),
    )
    return {
        'optimizer': optimizer,
        'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1},
    }


class BaseModule(pl.LightningModule):
    "Base Pytorch lightning module"
    def __init__(self, config, tokenizer=None):
        super().__init__()
        config = ExperimentConfig.from_dict(config)
        #Save hyperparameters
        self.save_hyperparameters({"config": asdict(config)})
        self.config = config
        self.tokenizer = tokenizer

    def forward(self, batch):
        raise NotImplementedError("Forward method not implemented!")

    def on_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()

    def on_after_backward(self):
        unused_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                unused_params.append(name)
        
        if unused_params:
            print(f"Parameters with no gradient: {unused_params}")

class Sub2StructModule(BaseModule):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        config = self.config
        gc = config.model.graph_encoder
        if config.data.dataset.substructure_featurizer == "morgan_counts":
            num_substructures = config.data.dataset.morgan_n_bits
        else:
            num_substructures = config.data.dataset.num_substructures

        self.model = Sub2Structure(
            tokenizer=self.tokenizer,
            structure_kwargs=config.model.structure_model,
            pretrained_structure_path=config.model.pretrained_structure_model_path,
            num_substructures=num_substructures,
            max_substructures_count=config.data.dataset.max_substructures_count,
            use_formula=config.model.use_formula,
            num_elements=config.data.dataset.num_elements,
            max_elements_count=config.data.dataset.max_elements_count,
            # Multi-view options
            use_graph=config.model.use_graph,
            use_substructures=config.model.use_substructures,
            gnn_pretrained_path=gc.pretrained_path,
            gnn_num_layer=gc.gnn_num_layer,
            gnn_emb_dim=gc.gnn_emb_dim,
            gnn_JK=gc.gnn_JK,
            gnn_drop_ratio=gc.gnn_drop_ratio,
            freeze_gnn=gc.freeze,
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=config.model.optimizer.label_smoothing
        )

    def training_step(self, batch, batch_idx):
        batch_size = len(batch['molecules'])
        outputs = self.model(batch)
        # Get loss
        loss, loss_info = self.compute_loss(outputs, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch['molecules'])
        outputs = self.model(batch)

        # Get loss
        loss, loss_info = self.compute_loss(outputs, batch)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        if 'structure_loss' in loss_info:
            self.log('val_structure_loss', loss_info['structure_loss'], prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # max_molecule_len = self.trainer.datamodule.val_dataset.max_molecule_len

        # Evaluate molecule generation (only for a few batches)
        # if self.trainer.current_epoch % 50 == 0 and batch_idx < 5:
        #     # Call compute_generation_metrics directly
        #     generation_metrics = self.model.compute_generation_metrics(
        #         structure_features=outputs['structure_features'],
        #         structure_features_mask=outputs['structure_features_mask'],
        #         true_smiles=batch['molecules'],
        #         max_molecule_len=max_molecule_len,
        #         max_samples=None,
        #         log_prefix="val"
        #     )
            
        #     # Log the metrics manually (since the metrics won't be logged by the method anymore)
        #     for name, value in generation_metrics.items():
        #         self.log(f'val_generation/{name}', value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss

    def compute_loss(self, outputs, batch, use_spectra=True):
        """
        Compute combined loss for multitask learning
        
        Args:
            outputs: Dictionary with model outputs
            batch: Dictionary with ground truth
            use_spectra: Whether we're in spectra mode (Stage 2) or substructure mode (Stage 1)
        """
        total_loss = 0.0
        loss_info = {}
        
        # Structure prediction loss (used in both stages)
        target = batch['target']
        
        # Reshape for loss calculation
        structure_logits = outputs['structure_logits'].view(-1, self.tokenizer.vocab_size)
        target = target.view(-1)
        
        structure_loss = self.loss_fn(structure_logits, target)
        total_loss += structure_loss
        loss_info['structure_loss'] = structure_loss.item()
        
        loss_info['total_loss'] = total_loss.item()
        return total_loss, loss_info
    
    def get_optimizer_params(self):
        """
        Get optimizers based on the training stage
        """
        opt_config = self.config.model.optimizer
        lr = opt_config.noam_factor if opt_config.use_noam_schedule else opt_config.lr
        return {
            'lr': lr,
            'betas': (0.9, 0.98),
            'eps': 1.0e-9,
            'weight_decay': 1.0e-5
        }

    def configure_optimizers(self):
        # Create optimizer
        opt_params = self.get_optimizer_params()
        optimizer = torch.optim.Adam(
            self.parameters(),
            **opt_params
        )
        return _build_optimizer_output(self, optimizer)


class Spec2StructModule(BaseModule):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        config = self.config
        self.model = Spectra2Structure(
            tokenizer=tokenizer,
            pretrained_spectrum_encoder_path= config.model.pretrained_spectrum_encoder_path,
            pretrained_structure_path=config.model.pretrained_structure_model_path,
            freeze_spectrum_encoder=False,
            multispectra_encoder_kwargs=(asdict(config.model.multispectra_encoder)
                                         if config.model.multispectra_encoder else None),
            use_tokenized_spectrum_encoder=config.model.use_tokenized_spectrum_encoder,
            tokenized_spectrum_encoder_kwargs=(asdict(config.model.tokenized_spectrum_encoder)
                                               if config.model.tokenized_spectrum_encoder else None),
            use_formula=config.model.use_formula,
            formula_baked_into_tokens=getattr(config.data.dataset, 'use_alberts_baseline_tokens', False),
            structure_model_kwargs=asdict(config.model.structure_model),
            label_smoothing=config.model.optimizer.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        # Get loss for spectra-to-structure stage
        batch_size = len(batch['molecules'])
        outputs = self.model(batch)
        loss, loss_info = self.model.compute_loss(outputs, batch) #this is pi_theta

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        for name, value in loss_info.items():
            self.log(f'train_{name}', value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get loss for spectra-to-structure stage
        batch_size = len(batch['molecules'])

        with torch.enable_grad():
            outputs = self.model(batch)

        loss, loss_info = self.model.compute_loss(outputs, batch)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        for name, value in loss_info.items():
            self.log(f'val_{name}', value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        max_molecule_len = self.trainer.datamodule.val_dataset.max_molecule_len
        # # Evaluate molecule generation (only for a few batches)
        # if (not self.trainer.sanity_checking) and batch_idx < 5:
        #     # Call compute_generation_metrics directly
        #     generation_metrics = self.model.compute_generation_metrics(
        #         structure_features=outputs['structure_features'],
        #         structure_features_mask=outputs['structure_features_mask'],
        #         true_smiles=batch['molecules'],
        #         max_molecule_len=max_molecule_len,
        #         max_samples=None,
        #         log_prefix="val"
        #     )
        #     # Log the metrics manually (since the metrics won't be logged by the method anymore)
        #     for name, value in generation_metrics.items():
        #         self.log(f'val_generation/{name}', value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        # If multitask calculate and log substructure-prediction metrics
        if hasattr(self.model, 'calculate_substructure_metrics'):
            metrics = self.model.calculate_substructure_metrics(outputs=outputs, batch=batch)
            for name, value in metrics.items():
                self.log(f'val_{name}', value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss

    def get_optimizer_params(self):
        """
        Get optimizers based on the training stage
        """
        opt_config = self.config.model.optimizer
        lr = opt_config.noam_factor if opt_config.use_noam_schedule else opt_config.lr
        # OpenNMT/Alberts Adam recipe: beta2=0.998, eps=1e-9, no weight decay.
        return {
            'lr': lr,
            'betas': (0.9, 0.999),
            'eps': 1.0e-8,
            'weight_decay': 0.0
        }

    def configure_optimizers(self):
        opt_params = self.get_optimizer_params()
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            **opt_params
        )
        return _build_optimizer_output(self, optimizer)
