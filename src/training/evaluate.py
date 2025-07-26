# Add this import at the top
from data.lipsync_dataset import LipSyncDataset, collate_fn
# ... existing imports ...

# Example usage (in a main function or script flow for evaluation):
if __name__ == "__main__":
    # Load config
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Evaluator
    evaluator = Evaluator(config, device)
    evaluator.load_models(config["paths"]["models"]) # Load trained models

    # Initialize Dataset and DataLoader for evaluation
    # Make sure 'data/processed' or 'data/raw' contains your video and audio pairs
    eval_dataset = LipSyncDataset(data_dir=config["paths"]["data_processed"], config_path="config.yaml", is_train=False)
    # Adjust batch_size as needed, usually larger for evaluation
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=4) # Assuming eval_batch_size in config

    logger.info("Starting Evaluation...")
    metrics = evaluator.evaluate(eval_dataloader)
    print("Evaluation Metrics:", metrics)