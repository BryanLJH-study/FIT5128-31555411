import os
import torch
import torch.nn as nn
import torch.optim as optim

from modules.utils import load_data, save_config, test_model
from modules.trainer_tester import TrainerTester
from models.mfcc_only.model import DepressionDetectionModel



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running Script: {script_dir}")

    config = {
        "DAIC_WOZ_READER": {
            "directory": "../data/preprocessed_features/DAIC-WOZ/MFCC_Isolated",
            "speaking_only": False,
            "keep_AU_cols": [],
            "phq8_categories": {
                "0": [0, 9],
                "1": [10, 24]
            }, 
            "remove_unsuccessful": False
        },
        "SplitMode": "official",
        "DataPipeline": {
            "mfcc_dir": "../data/preprocessed_features/DAIC-WOZ/MFCC_Isolated",
            "normalize_mfcc": True,
            "mfcc_fixed_length": None,
            "au_separate": False,
            "au_fixed_length": None,
            "segment_config": {
                "au_window_size": None,
                "au_step_size": None,
                "mfcc_window_size": int(60 * (1000/60)),
                "mfcc_step_size": int(60 * (1000/60)),
            },
            "include_gender": False,
            "batch_size": 64,
            "weighted_random_sampling": True
        },
        "Model": {
            "num_mfcc_features": 60,
            "mfcc_fixed_length": int(60 * (1000/60)),
            "hidden_size": 40
        },
        "Training": {
            "num_epochs": 15,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "checkpoint_dir": os.path.join(script_dir, "checkpoints"),
            "log_dir": os.path.join(script_dir, "logs")
        }
    }
    

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    dataloaders, split_dfs = load_data(config)

    # Model
    model = DepressionDetectionModel(**config["Model"])

    # Loss function
    cat_0 = 0
    cat_1 = 0
    for sample_no in range(len(dataloaders["train"].dataset)):
        if dataloaders["train"].dataset[sample_no]["Category"] == 0:
            cat_0 += 1
        else:
            cat_1 += 1
    pos_weight = None #torch.tensor((train_categories == 0).sum() / (train_categories == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["Training"]["learning_rate"], weight_decay=config["Training"]["weight_decay"])

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["Training"]["num_epochs"])

    # Input Adapter
    def input_adapter(batch: dict) -> dict:
        inputs = {}
        inputs['mfcc_input'] = batch['MFCCs'].to(device)
            
        return inputs


    # Save the configuration .
    config["Optimization"] = {
        "loss_function": str(type(criterion)), 
        "optimizer": str(type(optimizer)),
        "scheduler": str(type(scheduler)) 
    }
    save_config(config, config["Training"]["log_dir"])


    # Train Model
    trainer = TrainerTester(model=model, dataloaders=dataloaders, device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler, input_adapter=input_adapter)

    trainer.train(
        num_epochs=config["Training"]["num_epochs"],
        checkpoint_dir=config["Training"]["checkpoint_dir"],
        log_dir=config["Training"]["log_dir"]
    )

    # Test model
    test_model(config, trainer)


if __name__ == "__main__":
    main()
