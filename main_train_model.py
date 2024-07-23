#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import BertTokenizer, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from model_architecture import DataProcessor, CustomBertClassifier

class TextClassificationPipeline:
    """
    This class handles the entire pipeline for training, validating, and testing a BERT model 
    for text classification.

    Attributes:
    - dataset_path: str, path to the dataset CSV file.
    - model_name: str, name of the BERT model to use.
    - max_len: int, maximum length of the tokenized sequences.
    - batch_size: int, number of samples per batch for the DataLoader.
    - num_epochs: int, number of training epochs.
    - lr: float, learning rate for the optimizer.
    - freeze_bert_layers: int, number of BERT layers to freeze during training.
    - gamma: float, multiplicative factor of learning rate decay.
    - step_size: int, period of learning rate decay.
    - tokenizer: BertTokenizer, tokenizer for the BERT model.
    - data_processor: DataProcessor, processes the dataset.
    - train_dataset: Dataset, processed training dataset.
    - val_dataset: Dataset, processed validation dataset.
    - test_dataset: Dataset, processed test dataset.
    - train_dataloader: DataLoader, dataloader for the training dataset.
    - val_dataloader: DataLoader, dataloader for the validation dataset.
    - test_dataloader: DataLoader, dataloader for the test dataset.
    - model: CustomBertClassifier, the custom BERT model for classification.
    - device: torch.device, device to run the model on (CPU or GPU).
    - optimizer: AdamW, optimizer for the model.
    - scheduler: StepLR, learning rate scheduler.
    """
    def __init__(self, dataset_path, model_name='bert-base-uncased', max_len=256, batch_size=32, num_epochs=20, lr=1e-5, freeze_bert_layers=0, gamma=0.65, step_size=1):
        """
        Initializes the TextClassificationPipeline with dataset and model parameters, along with other configurations.
        
        Input:
        - dataset_path: str, path to the dataset CSV file.
        - model_name: str, name of the BERT model to use. Default is 'bert-base-uncased'.
        - max_len: int, maximum length of the tokenized sequences. Default is 256.
        - batch_size: int, number of samples per batch for the DataLoader. Default is 32.
        - num_epochs: int, number of training epochs. Default is 20.
        - lr: float, learning rate for the optimizer. Default is 1e-5.
        - freeze_bert_layers: int, number of BERT layers to freeze during training. Default is 0.
        - gamma: float, multiplicative factor of learning rate decay. Default is 0.65.
        - step_size: int, period of learning rate decay. Default is 1.
        
        Output:
        - Initializes the required components for the text classification pipeline.
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.freeze_bert_layers = freeze_bert_layers
        self.gamma = gamma
        self.step_size = step_size

        # Initialize tokenizer and data processor
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.data_processor = DataProcessor(self.dataset_path)

        # Get datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self.data_processor.get_datasets(self.tokenizer, self.max_len)

        # Initialize data loaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        self.model = CustomBertClassifier(num_labels=self.data_processor.y_train_one_hot.shape[1], freeze_bert_layers=self.freeze_bert_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer and learning rate scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def train(self):
        """
        Trains the BERT model using the training dataset and validates it using the validation dataset.
        Also saves the trained model and tests it after training.
        
        Input:
        - None (uses initialized attributes)
        
        Output:
        - Trains the model, validates it, saves the trained model, and tests it. Plots metrics as well.
        """
        avg_train_loss_list = []
        avg_val_loss_list = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for batch in self.train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                probabilities = self.model(input_ids, attention_mask)
                loss = self.model.compute_loss(probabilities, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                predicted_labels = torch.argmax(probabilities, dim=1)
                lab = torch.argmax(labels, dim=1)
                correct_predictions += (predicted_labels == lab).sum().item()
                total_predictions += len(labels)

            avg_train_loss = total_loss / len(self.train_dataloader)
            avg_train_loss_list.append(avg_train_loss)
            train_accuracies.append(correct_predictions / total_predictions)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Training Loss: {avg_train_loss}")

            self.validate(avg_val_loss_list, val_accuracies)

            self.scheduler.step()
            print('Epoch:', epoch, 'LR:', self.scheduler.get_last_lr())

        self.plot_metrics(avg_train_loss_list, avg_val_loss_list, train_accuracies, val_accuracies)

        # Save the trained model
        torch.save(self.model.state_dict(), 'trained_model.pth')
        print("Model saved to 'trained_model.pth'")

        # Test the model after training
        self.test()

    def validate(self, avg_val_loss_list, val_accuracies):
        """
        Validates the BERT model using the validation dataset and updates the lists with average validation loss 
        and accuracy for each epoch.
        
        Input:
        - avg_val_loss_list: list, stores the average validation loss for each epoch.
        - val_accuracies: list, stores the validation accuracy for each epoch.
        
        Output:
        - Updates avg_val_loss_list and val_accuracies with the validation metrics.
        """
        self.model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                probabilities = self.model(input_ids, attention_mask)
                val_loss = self.model.compute_loss(probabilities, labels)
                total_val_loss += val_loss.item()

                predicted_labels = torch.argmax(probabilities, dim=1)
                lab = torch.argmax(labels, dim=1)
                correct_predictions += (predicted_labels == lab).sum().item()
                total_predictions += len(labels)

        avg_val_loss = total_val_loss / len(self.val_dataloader)
        avg_val_loss_list.append(avg_val_loss)
        val_accuracies.append(correct_predictions / total_predictions)
        print(f"Average Validation Loss: {avg_val_loss}")

    def test(self):
        """
        Tests the BERT model using the test dataset and prints the accuracy.
        
        Input:
        - None (uses initialized attributes)
        
        Output:
        - Prints the accuracy of the model on the test dataset.
        """
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in self.test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                probabilities = self.model(input_ids, attention_mask)
                predicted_labels = torch.argmax(probabilities, dim=1)
                lab = torch.argmax(labels, dim=1)
                correct_predictions += (predicted_labels == lab).sum().item()
                total_predictions += len(labels)

        accuracy = correct_predictions / total_predictions
        print(f"Test Accuracy: {accuracy:.4f}")

    def plot_metrics(self, avg_train_loss_list, avg_val_loss_list, train_accuracies, val_accuracies):
        """
        Plots the training and validation loss and accuracy metrics.
        
        Input:
        - avg_train_loss_list: list, average training loss for each epoch.
        - avg_val_loss_list: list, average validation loss for each epoch.
        - train_accuracies: list, training accuracy for each epoch.
        - val_accuracies: list, validation accuracy for each epoch.
        
        Output:
        - Displays the plots for training and validation loss and accuracy.
        """
        plt.figure(figsize=(10, 5))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(avg_train_loss_list) + 1), avg_train_loss_list, label='Training Loss')
        plt.plot(range(1, len(avg_val_loss_list) + 1), avg_val_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    model_name = 'best'  # Change this to 'freezed' in order to train the freezed version

    # Initialize the pipeline with the dataset path and other parameters
    if model_name == 'best':
        pipeline = TextClassificationPipeline(
            dataset_path="extracted_data_filtered10k.csv",
            model_name='bert-base-uncased',
            max_len=256,
            batch_size=32,
            num_epochs=20,
            lr=1e-5,
            freeze_bert_layers=0,
            gamma=0.65,
            step_size=1
        )
    elif model_name == 'freezed':
        pipeline = TextClassificationPipeline(
            dataset_path="extracted_data_filtered10k.csv",
            model_name='bert-base-uncased',
            max_len=256,
            batch_size=32,
            num_epochs=30,
            lr=1e-5,
            freeze_bert_layers=12,
            gamma=0.85,
            step_size=1
        )
        
    # Start training the model
    pipeline.train()

