import pytorch_lightning as pl
import torch

class GradientFlowCallback(pl.Callback):
    def __init__(self, log_frequency=300):
        super().__init__()
        self.log_frequency = log_frequency
        
    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.log_frequency != 0 or pl_module.global_rank != 0:
            return
            
        # Check which components have gradients
        has_gradients = {}
        gradient_stats = {}
        
        # Find WandbLogger if it exists
        wandb_logger = None
        if hasattr(pl_module, "logger"):
            if isinstance(pl_module.logger, list):
                for logger in pl_module.logger:
                    if "WandbLogger" in str(type(logger)):
                        wandb_logger = logger
                        break
            elif "WandbLogger" in str(type(pl_module.logger)):
                wandb_logger = pl_module.logger
        
        # Check spectrum encoder gradients
        spectrum_grad = any(p.grad is not None for p in pl_module.model.spectrum_encoder.parameters())
        has_gradients['spectrum_encoder'] = spectrum_grad
        
        # Check cross-attention gradients
        cross_attn_grad = any(p.grad is not None for p in pl_module.model.cross_attention_layers.parameters())
        has_gradients['cross_attention'] = cross_attn_grad
        
        # Check T5 components
        t5_decoder_grad = any(p.grad is not None for p in pl_module.model.t5.decoder.parameters())
        has_gradients['t5_decoder'] = t5_decoder_grad
        
        t5_lm_head_grad = any(p.grad is not None for p in pl_module.model.t5.lm_head.parameters())
        has_gradients['t5_lm_head'] = t5_lm_head_grad
        
        # print(f"\n===== GRADIENT FLOW AT STEP {trainer.global_step} =====")
        # for k, v in has_gradients.items():
        #     print(f"  {k}: {'✓' if v else '✗'}")
        #     # Log to metrics
        #     gradient_stats[f"gradients/has_{k}"] = float(v)
            
        # Also check gradient norms for active components
        for name, flag in has_gradients.items():
            if flag:
                if name == 'spectrum_encoder':
                    params = pl_module.model.spectrum_encoder.parameters()
                elif name == 'cross_attention':
                    params = pl_module.model.cross_attention_layers.parameters()
                elif name == 't5_decoder':
                    params = pl_module.model.t5.decoder.parameters()
                elif name == 't5_lm_head':
                    params = pl_module.model.t5.lm_head.parameters()
                else:
                    continue
                    
                # Calculate average norm
                norms = []
                for p in params:
                    if p.grad is not None:
                        norms.append(p.grad.norm().item())
                        
                if norms:
                    avg_norm = sum(norms) / len(norms)
                    max_norm = max(norms)
                    print(f"    Avg grad norm for {name}: {avg_norm:.6f}")
                    print(f"    Max grad norm for {name}: {max_norm:.6f}")
                    
                    # Log to metrics
                    gradient_stats[f"gradients/{name}_mean"] = avg_norm
                    gradient_stats[f"gradients/{name}_max"] = max_norm
                    
                    # Sample histogram for W&B if available (for every 5th step)
                    if wandb_logger and trainer.global_step % (self.log_frequency * 5) == 0:
                        try:
                            import wandb
                            import numpy as np
                            
                            # Create histogram of gradient norms
                            wandb_logger.experiment.log({
                                f"gradient_histogram/{name}": wandb.Histogram(np.array(norms)),
                                "global_step": trainer.global_step
                            })
                            
                            # Optional: Log parameter update ratio (gradient norm / parameter norm)
                            # This helps identify if updates are too small compared to parameter values
                            param_norms = [p.norm().item() for p in params if p.requires_grad]
                            if param_norms:
                                update_ratios = [g/max(p, 1e-7) for g, p in zip(norms, param_norms)]
                                wandb_logger.experiment.log({
                                    f"parameter_update_ratio/{name}": wandb.Histogram(np.array(update_ratios)),
                                    "global_step": trainer.global_step
                                })
                                
                        except Exception as e:
                            print(f"Error logging gradient histogram: {e}")
        
        # Log all stats to the trainer for consistency
        pl_module.log_dict(gradient_stats)