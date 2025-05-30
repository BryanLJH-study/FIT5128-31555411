import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix



class TrainerTester:
    def __init__(
        self,
        model: nn.Module,
        dataloaders: dict,
        device: torch.device,
        criterion: nn.Module,
        optimizer: optim,
        scheduler: optim.lr_scheduler = None,
        input_adapter=None                    # Function that maps a batch to model inputs.
    ):
        """
        Args:
            - model (nn.Module): The PyTorch model.
            - dataloaders (dict): Dictionary with keys 'train', 'val', and optionally 'test'.
            - device (torch.device): Device to run on.
            - criterion (nn.Module): Pytorch loss function to use
            - optimizer (optim): Pytorch optimizer to use
            - scheduler (optim.lr_scheduler, optional): Pytorch scheduler to use
            - input_adapter (callable, optional): A function that accepts a batch (dict) and returns a dictionary of
                                                  keyword arguments to pass to the model. If None, a default adapter is used.
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Input adapter: a function to convert a batch dictionary to model inputs.
        if input_adapter is None:
            self.input_adapter = self.default_input_adapter
        else:
            self.input_adapter = input_adapter


    def default_input_adapter(self, batch: dict) -> dict:
        """
        Default adapter that extracts keys from the batch and returns them as keyword arguments.
        Note: These keys may or may not be present.
        """
        inputs = {}

        if 'AUs' in batch:
            inputs['au_input'] = batch['AUs'].to(self.device)
        if 'AU_r' in batch:
            inputs['au_r_input'] = batch['AU_r'].to(self.device)
        if 'AU_c' in batch:
            inputs['au_c_input'] = batch['AU_c'].to(self.device)
        if 'MFCCs' in batch:
            inputs['mfcc_input'] = batch['MFCCs'].to(self.device)
        if 'Gender' in batch:
            inputs['gender_input'] = batch['Gender'].to(self.device)
        if 'AU_lengths' in batch:
            inputs['au_lengths'] = batch['AU_lengths']
        if 'MFCCs_lengths' in batch:
            inputs['mfccs_lengths'] = batch['MFCCs_lengths']
            
        return inputs


    def load_checkpoint(self, path: str):
        """
        Load previously saved model (to be used in testing).

        Args
         - path (str): Directory to saved model
        """
        checkpoint = torch.load(path, map_location=self.device)

        # If the checkpoint is a dict containing optimizer state, etc.
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Loaded model states from {path}")


    def train(self, num_epochs: int, checkpoint_dir: str, log_dir: str = "./logs", resume_from_checkpoint: str = None):
        """
        Trains the model.

        Args:
            - num_epochs (int): Number of training epochs.
            - checkpoint_dir (str): Directory to save checkpoints.
            - log_dir (str): Directory to save training logs.
            - resume_from_checkpoint (str, optional): Path to checkpoint file to resume from.
        """
        # Initialize checkpoint & log directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Intialize variables
        start_epoch = 0
        train_metrics = {'loss': [], 'accuracy': []}
        val_metrics = {'loss': [], 'accuracy': []}
        best_val_loss = float('inf')
        best_val_acc = 0.0

        # load data from checkpoint (if specified)
        if resume_from_checkpoint is not None:
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed training from epoch {start_epoch}")

        # Train model
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                # For each batch
                for batch in tqdm(self.dataloaders[phase], desc=phase.capitalize()):
                    # Use the input adapter to prepare model inputs.
                    model_inputs = self.input_adapter(batch)

                    # Forward pass
                    outputs, attention_weights = self.model(**model_inputs)
                    labels = batch["Category"].to(self.device)

                    if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                        outputs = outputs.squeeze(dim=1)
                        labels = labels.float()

                    loss = self.criterion(outputs, labels)

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * labels.size(0)
                    total_samples += labels.size(0)

                    # Compute predictions and update accuracy.
                    if isinstance(self.criterion, nn.CrossEntropyLoss):
                        _, preds = torch.max(outputs, 1)

                    elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                        preds = (torch.sigmoid(outputs) > 0.5).long()
                    
                    correct_predictions += (preds == labels).sum().item()

                epoch_loss = running_loss / total_samples
                epoch_accuracy = correct_predictions / total_samples
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Update metrics logs
                if phase == 'train':
                    train_metrics['loss'].append(epoch_loss)
                    train_metrics['accuracy'].append(epoch_accuracy)
                else:
                    val_metrics['loss'].append(epoch_loss)
                    val_metrics['accuracy'].append(epoch_accuracy)

                # Update scheduler (if provided)
                if phase == "val" and self.scheduler is not None:
                    print(f"learning rate: {self.scheduler.get_last_lr()}")
                    
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'][-1])
                    else:
                        self.scheduler.step()
                   
            # Save checkpoint each epoch.
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)

            # If lowest validation loss, update best model
            if epoch > 1 and val_metrics['loss'][-1] < best_val_loss:
                best_val_loss = val_metrics['loss'][-1]
                best_model_path = os.path.join(checkpoint_dir, "best_loss_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print("Best loss model updated.")

            # If highest validation accuracy, update best model
            if epoch > 1 and val_metrics["accuracy"][-1] > best_val_acc:
                best_val_acc = val_metrics['accuracy'][-1]
                best_model_path = os.path.join(checkpoint_dir, "best_acc_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print("Best accuracy model updated.")


        # Save training metrics
        with open(os.path.join(log_dir, "train_metrics.json"), "w") as f:
            json.dump(train_metrics, f)
        with open(os.path.join(log_dir, "val_metrics.json"), "w") as f:
            json.dump(val_metrics, f)

        print("Training complete.")



    def test_predictions(self, checkpoint_path: str = None):
        """
        Collect predictions and true labels from the test dataset.
        
        Args:
            - checkpoint_path (str, optional): Path to a checkpoint file to load the model.
        
        Returns:
            - predictions and true labels.
        """
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dataloaders["test"]:
                model_inputs = self.input_adapter(batch)
                outputs, attention_weights = self.model(**model_inputs)

                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    outputs = outputs.squeeze(dim=1)
                    preds = (torch.sigmoid(outputs) > 0.5).long()

                elif isinstance(self.criterion, nn.CrossEntropyLoss):
                    _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["Category"].cpu().numpy())

        return all_preds, all_labels
    


    def test(self, checkpoint_path: str = None):
        """
        Evaluate the model on the test dataset. Optionally load a checkpoint before testing.
        
        Args:
            - checkpoint_path (str, optional): Path to a checkpoint file.

        Returns:
            - dict: A dictionary containing the confusion matrix and classification report.
        """
        preds, labels = self.test_predictions(checkpoint_path)

        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds)

        return {"confusion_matrix": cm, "classification_report": report}
