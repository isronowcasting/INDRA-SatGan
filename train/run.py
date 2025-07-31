import os
import sys
from pathlib import Path
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Set WandB offline mode and GPU
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dgmr import DGMR
from dataset import DGMRDataModule
from EvaluationCallback import EvaluationCallback

# --- Main Training Logic ---
if __name__ == '__main__':
    # --- Configuration ---
    FORECAST_STEPS = 6
    INPUT_FRAMES = 6
    
    checkpoint_dir = "/home/sac/data_67/Nowcasting/vinayak/ModelSouth/train/checkpoints_full"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_files = sorted(Path(checkpoint_dir).glob("*.ckpt"), key=os.path.getmtime)
    resume_checkpoint = str(ckpt_files[-1]) if ckpt_files else None
    
    if resume_checkpoint: print(f"Resuming from checkpoint: {resume_checkpoint}")
    else: print("No checkpoint found, starting fresh.")

    wandb.init(project="dgmr_final_eval", mode="offline")
    wandb_logger = WandbLogger(project="dgmr_final_eval", mode="offline")
    
    # --- Instantiate Callbacks ---
    eval_callback = EvaluationCallback(
        save_dir=checkpoint_dir,
        log_file=os.path.join(checkpoint_dir, "evaluation_summary.log"),
        csv_log_file="evaluation_metrics.csv"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, 
        filename="epoch{epoch:02d}", 
        save_top_k=-1, 
        every_n_epochs=1
    )

    trainer = Trainer(
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, eval_callback],
        accelerator="auto",
        precision=32,
        log_every_n_steps=1,
    )

    model = DGMR(
        visualize=True, 
        forecast_steps=FORECAST_STEPS,
        output_shape=128,
        input_frames=INPUT_FRAMES,
        grid_lambda=2
    )

    datamodule = DGMRDataModule(
        dataset_folder="/home/sac/data_67/Nowcasting/data/required_south",
        csv_path="/home/sac/data_67/Nowcasting/data/input_sequences_south.csv",
        val_split=0.05,
        num_workers=4,
        num_input_frames=INPUT_FRAMES,
        num_target_frames=FORECAST_STEPS,
        batch_size=16
    )
    
    trainer.fit(model, datamodule, ckpt_path=resume_checkpoint)