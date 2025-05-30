import os
import json
import pandas as pd
import numpy as np

from modules.daic_woz_reader import DAIC_WOZ_READER
from modules.data_pipeline import DepressionDataPipeline
from modules.trainer_tester import TrainerTester


def load_config(config_path: str) -> dict:
    """
    Load configuration parameters from a JSON file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_config(config: dict, output_dir: str):
    """
    Save configuration parameters to a JSON file in the given directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nConfiguration saved to {config_path}")


def load_data(config: dict):
    """
    Load and preprocess data.
    """
    print("Reading DAIC-WOZ Data")
    facial_data = DAIC_WOZ_READER(**config["DAIC_WOZ_READER"])
    facial_data.au_df_stats()  

    au_df = facial_data.au_df if len(config["DAIC_WOZ_READER"]["keep_AU_cols"]) > 0 else None

    if config["SplitMode"] == "official":
        split_dfs = facial_data.split_dfs

    elif config["SplitMode"] == "80-10-10":
        df = pd.concat((facial_data.split_dfs["train"], facial_data.split_dfs["val"], facial_data.split_dfs["test"]))
        df = df.sample(frac=1, random_state=12345)

        df_0 = df[df["Category"] == 0]
        train_0, val_0, test_0 = np.split(df_0, [int(0.8*len(df_0)), int(0.9*len(df_0))])

        df_1 = df[df["Category"] == 1]
        train_1, val_1, test_1 = np.split(df_1, [int(0.8*len(df_1)), int(0.9*len(df_1))])

        split_dfs = {
            "train": pd.concat([train_0, train_1]),
            "val": pd.concat([val_0, val_1]),
            "test": pd.concat([test_0, test_1]),
}
            
    print("\nPreparing Dataloader")
    data_pipeline = DepressionDataPipeline(split_df=split_dfs, au_df=au_df, **config["DataPipeline"])

    return data_pipeline.dataloaders, split_dfs


def test_model(config: dict, trainer: TrainerTester):
    # Test Final Epoch model
    print("\nTESTING FINAL MODEL")
    test_metrics = trainer.test(None)
    print("\nConfusion Matrix\n", test_metrics["confusion_matrix"])
    print("\nClassification Report\n", test_metrics["classification_report"])

    # Save testing metrics
    np.save(os.path.join(config["Training"]["log_dir"], "final_model_test_confusion_matrix.npy"), test_metrics["confusion_matrix"])


    # Test Best Loss Model
    print("\nTESTING BEST LOSS MODEL")
    test_metrics = trainer.test(os.path.join(config["Training"]["checkpoint_dir"], "best_loss_model.pth"))
    print("\nConfusion Matrix\n", test_metrics["confusion_matrix"])
    print("\nClassification Report\n", test_metrics["classification_report"])

    # Save testing metrics
    np.save(os.path.join(config["Training"]["log_dir"], "best_loss_test_confusion_matrix.npy"), test_metrics["confusion_matrix"])


    # Test Best Accuracy Model
    print("\nTESTING BEST ACCURACY MODEL")
    test_metrics = trainer.test(os.path.join(config["Training"]["checkpoint_dir"], "best_acc_model.pth"))
    print("\nConfusion Matrix\n", test_metrics["confusion_matrix"])
    print("\nClassification Report\n", test_metrics["classification_report"])

    # Save testing metrics
    np.save(os.path.join(config["Training"]["log_dir"], "best_acc_test_confusion_matrix.npy"), test_metrics["confusion_matrix"])