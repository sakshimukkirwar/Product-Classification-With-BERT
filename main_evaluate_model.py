#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from model_architecture import DataProcessor, CustomBertClassifier

class ModelEvaluator:
    """
    This class is responsible for evaluating a pre-trained BERT model on a given dataset.
    
    Attributes:
    - dataset_path: str, path to the dataset CSV file.
    - model_path: str, path to the pre-trained model file.
    - model_name: str, name of the BERT model to use.
    - max_len: int, maximum length of the tokenized sequences.
    - batch_size: int, number of samples per batch for the DataLoader.
    - tokenizer: BertTokenizer, tokenizer for the BERT model.
    - data_processor: DataProcessor, processes the dataset.
    - test_dataset: Dataset, processed test dataset.
    - test_dataloader: DataLoader, dataloader for the test dataset.
    - model: CustomBertClassifier, the custom BERT model for classification.
    - device: torch.device, device to run the model on (CPU or GPU).
    """
    def __init__(self, dataset_path, model_path, model_name='bert-base-uncased', max_len=256, batch_size=32):
        """
        Initializes the ModelEvaluator with dataset and model paths, along with other configurations.
        
        Input:
        - dataset_path: str, path to the dataset CSV file.
        - model_path: str, path to the pre-trained model file.
        - model_name: str, name of the BERT model to use. Default is 'bert-base-uncased'.
        - max_len: int, maximum length of the tokenized sequences. Default is 256.
        - batch_size: int, number of samples per batch for the DataLoader. Default is 32.
        
        Output:
        - Initializes the required components for model evaluation.
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size

        # Initialize tokenizer and data processor
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.data_processor = DataProcessor(self.dataset_path)

        # Get test dataset
        _, _, self.test_dataset = self.data_processor.get_datasets(self.tokenizer, self.max_len)

        # Initialize data loader
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        self.model = CustomBertClassifier(num_labels=self.data_processor.y_train_one_hot.shape[1])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)

    def evaluate(self):
        """
        Evaluates the model on the test dataset and calculates the accuracy.
        
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


if __name__ == '__main__':
    name = 'best'  # Change to 'freezed' for evaluating the frozen model

    if name == 'best':
        evaluator = ModelEvaluator(
            dataset_path="extracted_data_filtered10k.csv",
            model_path="model_project10k-gamm65.pth",
            model_name='bert-base-uncased',
            max_len=256,
            batch_size=32
        )
    elif name == 'freezed':
        # Initialize the evaluator with the dataset path and model file path
        evaluator = ModelEvaluator(
            dataset_path="extracted_data_filtered10k.csv",
            model_path="model_project10k-gamm65-12Freeze.pth",
            model_name='bert-base-uncased',
            max_len=256,
            batch_size=32
        )
    # Evaluate the model on the test data
    evaluator.evaluate()

