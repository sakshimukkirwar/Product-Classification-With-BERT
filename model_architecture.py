#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.utils import to_categorical
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from torch import nn
import torch.nn.functional as F

# Static class to clean text by removing stop words and non-alphabet characters
class TextCleaner:
    """
    Static class to clean text by removing stop words and non-alphabet characters.
    """
    stopWords = {'a','about','above','after','again','against','ain','all','am','an','and','any','are','aren',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d','did','didn',"didn't",'do','does','doesn',"doesn't",'doing','don',"don't",'down','during','each','few','for','from','further','had','hadn',"hadn't",'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn',"needn't",'no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who','whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves'}

    @staticmethod
    def clean_text(text):
        """
        Clean the input text by removing stop words and non-alphabet characters.

        Args:
            text (str): Input text string to be cleaned.

        Returns:
            str: Cleaned text string with stop words and non-alphabet characters removed.
        """
        text = text.lower() 
        text = re.sub("[^a-z]", " ", text)
        words = [word for word in text.split() if word not in TextCleaner.stopWords]
        return " ".join(words)


# Class to handle data loading, cleaning, and splitting
class DataProcessor:
    """
    Class to handle data loading, cleaning, and splitting.
    """
    def __init__(self, dataset_path):
        """
        Initialize the DataProcessor with a dataset path.

        Args:
            dataset_path (str): Path to the CSV dataset file.
        """
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path, sep='\u0007')
        
        self._prepare_data()
        self._encode_labels()

    def _prepare_data(self):
        """
        Prepare the data by cleaning text, handling missing values, and splitting into train, validation, and test sets.
        """
        mask = self.df['Title'].notna() & self.df['description'].notna()
        self.df['title_descr'] = self.df['Title'].astype(str) + ' ' + self.df['description'].astype(str)
        self.df = self.df[mask]
        self.df["title_descr"] = self.df.apply(lambda row: TextCleaner.clean_text(row["title_descr"]), axis=1)
        self.df = self.df[~self.df['Category'].isna()]

        train_df, temp_df = train_test_split(self.df, test_size=0.3, random_state=42)
        self.train_texts = train_df['title_descr'].tolist()
        self.train_labels = train_df['Category'].tolist()
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        self.val_texts = val_df['title_descr'].tolist()
        self.val_labels = val_df['Category'].tolist()
        self.test_texts = test_df['title_descr'].tolist()
        self.test_labels = test_df['Category'].tolist()

    def _encode_labels(self):
        """
        Encode the labels into integer and one-hot encoded formats.
        """
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(self.train_labels)
        self.y_val_encoded = self.label_encoder.transform(self.val_labels)
        self.y_test_encoded = self.label_encoder.transform(self.test_labels)

        self.y_train_one_hot = to_categorical(self.y_train_encoded)
        self.y_val_one_hot = to_categorical(self.y_val_encoded)
        self.y_test_one_hot = to_categorical(self.y_test_encoded)

    def get_datasets(self, tokenizer, max_len):
        """
        Get the datasets for training, validation, and testing.

        Args:
            tokenizer (BertTokenizer): Tokenizer to convert text to input IDs for BERT.
            max_len (int): Maximum length of the tokenized input sequences.

        Returns:
            tuple: Tuple containing the train, validation, and test datasets.
        """
        train_dataset = TextClassificationDataset(self.train_texts, self.y_train_one_hot, tokenizer, max_len)
        val_dataset = TextClassificationDataset(self.val_texts, self.y_val_one_hot, tokenizer, max_len)
        test_dataset = TextClassificationDataset(self.test_texts, self.y_test_one_hot, tokenizer, max_len)
        return train_dataset, val_dataset, test_dataset


# Custom dataset class to handle tokenization and data preparation for BERT
class TextClassificationDataset(Dataset):
    """
    Custom dataset class to handle tokenization and data preparation for BERT.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Initialize the dataset with texts, labels, tokenizer, and maximum sequence length.

        Args:
            texts (list): List of text samples.
            labels (list): List of corresponding labels for the text samples.
            tokenizer (BertTokenizer): Tokenizer to convert text to input IDs for BERT.
            max_len (int): Maximum length of the tokenized input sequences.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing tokenized input IDs, attention mask, and labels.
        """
        text = self.texts[index]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_len, padding='max_length', truncation=True)
        label = self.labels[index]
        return {'input_ids': inputs['input_ids'].flatten(), 'attention_mask': inputs['attention_mask'].flatten(), 'labels': torch.tensor(label)}


# Custom BERT model with additional layers for classification
class CustomBertClassifier(nn.Module):
    """
    Custom BERT model with additional layers for classification.
    """
    def __init__(self, num_labels, freeze_bert_layers=0):
        """
        Initialize the BERT classifier with the number of labels and optional layer freezing.

        Args:
            num_labels (int): Number of output labels for classification.
            freeze_bert_layers (int, optional): Number of initial BERT layers to freeze during training. Defaults to 0.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.num_labels = num_labels

        # Freeze BERT layers if specified
        if (freeze_bert_layers > 0):
            for param in list(self.bert.parameters())[:freeze_bert_layers]:
                param.requires_grad = False

        self.dropout = nn.Dropout(0.2)
        self.hidden_layer1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2)
        self.batch_norm = nn.BatchNorm1d(self.bert.config.hidden_size // 2)
        self.hidden_layer2 = nn.Linear(self.bert.config.hidden_size // 2, self.bert.config.hidden_size // 4)
        self.hidden_layer3 = nn.Linear(self.bert.config.hidden_size // 4, self.num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask to avoid attending to padding tokens.

        Returns:
            torch.Tensor: Probabilities for each class.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs["last_hidden_state"][:, 0, :]
        hidden_output = self.hidden_layer1(pooled_output)
        hidden_output = self.batch_norm(hidden_output)
        hidden_output = F.leaky_relu(hidden_output)
        hidden_output = self.hidden_layer2(hidden_output)
        hidden_output = F.leaky_relu(hidden_output)
        logits = self.hidden_layer3(hidden_output)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def compute_loss(self, probabilities, labels):
        """
        Compute the loss given the probabilities and labels.

        Args:
            probabilities (torch.Tensor): Predicted probabilities for each class.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(probabilities, labels)
        return loss

