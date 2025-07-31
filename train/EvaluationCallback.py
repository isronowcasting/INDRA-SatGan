import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import pandas as pd
import os
import numpy as np
import warnings

# Use the custom numpy-based CRPS function
from evaluation_utils import csi, fss, crps, denormalize_data, visualize_ensemble_grid, visualize_step_grid
from sklearn.metrics import mean_absolute_error

class EvaluationCallback(pl.Callback):
    def __init__(self, save_dir, log_file="evaluation_summary.log", csv_log_file="evaluation_metrics.csv"):
        super().__init__()
        self.save_dir = save_dir
        self.log_file_path = os.path.join(save_dir, log_file)
        self.csv_log_path = os.path.join(save_dir, csv_log_file)
        self.validation_step_outputs = []
        
        if not os.path.exists(self.csv_log_path):
            self._init_csv()

    def _init_csv(self):
        lead_times = ["30-min", "60-min", "90-min", "120-min", "150-min", "180-min"]
        thresholds = ["0.5", "2.5", "7.6", "10.0", "16", "25.0", "50"]
        
        headers = ['epoch']
        for lead in lead_times:
            headers.append(f"pred_MAE_{lead}")
            headers.append(f"pred_CRPS_{lead}")
            for t in thresholds:
                headers.append(f"pred_CSI@{t}_{lead}")
                headers.append(f"pred_FSS@{t}_{lead}")
            headers.append(f"pers_MAE_{lead}")
            for t in thresholds:
                headers.append(f"pers_CSI@{t}_{lead}")
                headers.append(f"pers_FSS@{t}_{lead}")
        
        pd.DataFrame(columns=headers).to_csv(self.csv_log_path, index=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        images, future_images, mean_prediction, ensemble_tensor = outputs

        phys_input = denormalize_data(images.cpu().numpy())
        phys_target = denormalize_data(future_images.cpu().numpy())
        phys_mean_pred = denormalize_data(mean_prediction.cpu().numpy())
        phys_ensemble = denormalize_data(ensemble_tensor.cpu().numpy())

        lead_time_steps = {"30-min": 0, "60-min": 1, "90-min": 2, "120-min": 3, "150-min": 4, "180-min": 5}
        physical_thresholds = {"0.5":0.5, "2.5":2.5, "7.6":7.6, "10.0":10, "16":16, "25.0":25, "50":50}
        
        batch_metrics = {}
        last_input_frame = phys_input[:, -1, 0, :, :]

        phys_target = phys_target.squeeze(axis=2)
        phys_mean_pred = phys_mean_pred.squeeze(axis=2)
        phys_ensemble = phys_ensemble.squeeze(axis=3)

        for lead_str, lead_idx in lead_time_steps.items():
            if lead_idx >= pl_module.hparams.get('forecast_steps', 6): continue

            target_frame = phys_target[:, lead_idx, :, :]
            mean_pred_frame = phys_mean_pred[:, lead_idx, :, :]
            ensemble_frame = phys_ensemble[:, :, lead_idx, :, :]

            batch_size = target_frame.shape[0]
            crps_scores = [crps(ensemble_frame[:, i, :, :], target_frame[i, :, :]) for i in range(batch_size)]
            
            batch_metrics[f"pred_MAE_{lead_str}"] = mean_absolute_error(target_frame.flatten(), mean_pred_frame.flatten())
            batch_metrics[f"pred_CRPS_{lead_str}"] = np.mean(crps_scores)
            
            batch_metrics[f"pers_MAE_{lead_str}"] = mean_absolute_error(target_frame.flatten(), last_input_frame.flatten())

            for t_str, t_val in physical_thresholds.items():
                batch_metrics[f"pred_CSI@{t_str}_{lead_str}"] = csi(mean_pred_frame, target_frame, threshold=t_val)
                batch_metrics[f"pred_FSS@{t_str}_{lead_str}"] = fss(mean_pred_frame, target_frame, threshold=t_val)
                batch_metrics[f"pers_CSI@{t_str}_{lead_str}"] = csi(last_input_frame, target_frame, threshold=t_val)
                batch_metrics[f"pers_FSS@{t_str}_{lead_str}"] = fss(last_input_frame, target_frame, threshold=t_val)
        
        self.validation_step_outputs.append(batch_metrics)
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.validation_step_outputs: return
        
        avg_metrics = {key: np.mean([d[key] for d in self.validation_step_outputs]) for key in self.validation_step_outputs[0]}
        
        pl_module.log_dict({f"val_epoch_avg/{k}": v for k, v in avg_metrics.items()}, prog_bar=False)
        self._log_summary(trainer.current_epoch, avg_metrics)
        self._log_to_csv(trainer.current_epoch, avg_metrics)
        
        # Use .get() for robust access to hparams
        if pl_module.hparams.get('visualize', False):
            self._generate_visualizations(trainer, pl_module)

        self.validation_step_outputs.clear()

    def _generate_visualizations(self, trainer, pl_module):
        print(f"\n--- Generating visualizations for epoch {trainer.current_epoch} ---")
        pl_module.eval()
        val_loader = trainer.datamodule.val_dataloader()
        try:
            # Get a single batch from the dataloader
            batch = next(iter(val_loader))

            # ** THE FIX IS HERE **
            # Manually move the batch to the same device as the model.
            # pl_module.device ensures this works on both CPU and GPU.
            images, future_images = batch
            images = images.to(pl_module.device)
            future_images = future_images.to(pl_module.device)
            
            # The batch is now on the correct device, so this call will work.
            with torch.no_grad():
                # Re-create the batch tuple to pass to validation_step
                gpu_batch = (images, future_images)
                _, _, mean_prediction, ensemble_tensor = pl_module.validation_step(gpu_batch, 0)
            
            step_identifier = trainer.global_step

            print("  - Generating mean prediction visualization...")
            visualize_step_grid(
                x=images, y=future_images, y_hat=mean_prediction,
                epoch=trainer.current_epoch, step=step_identifier, save_dir=self.save_dir
            )

            print("  - Generating ensemble visualization...")
            visualize_ensemble_grid(
                x=images, y=future_images, ensemble=ensemble_tensor,
                epoch=trainer.current_epoch, step=step_identifier, save_dir=self.save_dir
            )
            print("--- Visualizations saved ---")

        except Exception as e:
            warnings.warn(f"\nAn error occurred during visualization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always set the model back to training mode
            pl_module.train()

    def _log_summary(self, epoch, avg_metrics):
        summary_str = f"\n--- Epoch {epoch} Validation Summary ---\n"
        summary_str += pd.Series(avg_metrics).to_string()
        summary_str += f"\n{'-'*40}\n"
        print(summary_str)
        with open(self.log_file_path, "a") as f:
            f.write(summary_str)
            
    def _log_to_csv(self, epoch, avg_metrics):
        row_data = {'epoch': epoch}
        row_data.update(avg_metrics)
        df = pd.DataFrame([row_data])
        df.to_csv(self.csv_log_path, mode='a', header=not os.path.exists(self.csv_log_path), index=False)