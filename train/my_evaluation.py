import os
import sys
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# --- ENVIRONMENT SETUP ---
os.environ["WANDB_MODE"] = "disabled"  # No need to log to WandB for evaluation
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dgmr import DGMR
from dataset import DGMRDataModule
from EvaluationCallback import EvaluationCallback


# --- MAIN EVALUATION LOGIC ---
if __name__ == '__main__':
    # --- CORE CONFIGURATION ---
    # These parameters must match the model architecture saved in the checkpoint
    FORECAST_STEPS = 6
    INPUT_FRAMES = 6
    OUTPUT_SHAPE = 128
    BATCH_SIZE = 256 # You can increase this for faster evaluation if GPU memory allows
    
    # --- PATH CONFIGURATION ---
    checkpoint_dir = "/home/sac/data_67/Nowcasting/vinayak/ModelSouth/train/checkpoints_full"
    evaluation_output_dir = "/home/sac/data_67/Nowcasting/vinayak/ModelSouth/train/evaluation_results"
    os.makedirs(evaluation_output_dir, exist_ok=True)
    
    # Path to the specific checkpoint you want to evaluate
    checkpoint_to_evaluate = os.path.join(checkpoint_dir, "epochepoch=137.ckpt")
    
    if not os.path.exists(checkpoint_to_evaluate):
        print(f"ERROR: Checkpoint to evaluate not found at {checkpoint_to_evaluate}")
        sys.exit(1)

    # --- DATAMODULE CONFIGURATION FOR 100% VALIDATION ---
    datamodule = DGMRDataModule(
        dataset_folder="/home/sac/data_67/Nowcasting/data/required_valid_south",
        csv_path="/home/sac/data_67/Nowcasting/data/input_sequences_south_JJAS_strict.csv",
        val_split=1.0,  # CRITICAL: Use 100% of the data for the validation set
        num_workers=20,
        num_input_frames=INPUT_FRAMES,
        num_target_frames=FORECAST_STEPS,
        batch_size=BATCH_SIZE
    )

    # --- MODEL LOADING ---
    print(f"Loading model from checkpoint: {checkpoint_to_evaluate}")
    # Load the model directly from the checkpoint file.
    # We pass the hparams to ensure the architecture is constructed correctly before loading weights.
    model = DGMR.load_from_checkpoint(
        checkpoint_path=checkpoint_to_evaluate,
        strict=False, # Important for architecturally changed models
        # Provide any necessary hparams to reconstruct the model correctly
        output_shape=OUTPUT_SHAPE,
        forecast_steps=FORECAST_STEPS,
        input_frames=INPUT_FRAMES,
    )
    model.eval() # Set the model to evaluation mode
    print("Model loaded successfully.")

    # --- CALLBACKS ---
    # We use the same callback, but point its output to a new directory
    eval_callback = EvaluationCallback(
        save_dir=evaluation_output_dir,
        log_file=os.path.join(evaluation_output_dir, "final_evaluation_summary.log"),
        csv_log_file="final_evaluation_metrics.csv"
    )

    # --- TRAINER CONFIGURATION FOR VALIDATION ---
    # The trainer is just a runner here; no training will happen.
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision=32,
        logger=False, # No need for WandB or TensorBoard loggers
        callbacks=[eval_callback]
    )
    
    # --- RUN VALIDATION ---
    print("Starting evaluation on the full dataset...")
    # The `validate` method runs one full epoch over the validation dataloader
    trainer.validate(model, datamodule=datamodule)
    print("Evaluation complete.")
    print(f"Results saved in: {evaluation_output_dir}")