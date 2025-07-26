# Add this import at the top
from data.lipsync_dataset import LipSyncDataset, collate_fn
# ... existing imports ...

# Inside Trainer class or main training script, update how DataLoader is created:
# Example usage (in a main function or script flow):
if __name__ == "__main__":
    # Load config
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Trainer
    trainer = Trainer(config, device)

    # Initialize Dataset and DataLoader
    # Make sure 'data/processed' or 'data/raw' contains your video and audio pairs
    train_dataset = LipSyncDataset(data_dir=config["paths"]["data_processed"], config_path="config.yaml", is_train=True)
    # Adjust batch_size as needed
    train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    # You might also have a validation dataloader
    # val_dataset = LipSyncDataset(data_dir=config["paths"]["data_processed_val"], config_path="config.yaml", is_train=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=4)

    num_epochs = config["training"]["num_epochs"] # Assuming you add this to config.yaml
    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch+1}/{num_epochs}")
        avg_loss = trainer.train_epoch(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Optional: Save models periodically
        if (epoch + 1) % config["training"]["save_interval"] == 0: # Assuming save_interval in config
             trainer.save_models(config["paths"]["models"])

    trainer.save_models(config["paths"]["models"]) # Save final models