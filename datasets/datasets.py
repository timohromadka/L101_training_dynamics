# non-adversarial
# QNLI, WinoGrande
# adversarial 
# Adversarial Squad, WSC

from datasets import load_dataset

def load_dataset_with_splits(dataset_name):
    """
    Loads specified dataset with its train, validation, and test splits from Hugging Face's dataset library.

    Args:
    dataset_name (str): The name of the dataset to load.

    Returns:
    dict: A dictionary containing the 'train', 'validation', and 'test' splits of the dataset.
    """
    # Dataset split mapping
    split_mapping = {
        'qnli': ('train', 'validation', 'test'),
        'adversarial_qa:adversarialQA': ('train', 'validation', 'test'),
        'winogrande': ('train', 'validation', 'test'),
        'super_glue': ('train', 'validation', 'test') # For the WSC task
    }

    if dataset_name not in split_mapping:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    # Load the specified splits
    dataset_splits = split_mapping[dataset_name]
    dataset = load_dataset(dataset_name, split=list(dataset_splits))

    return {split: dataset[split] for split in dataset_splits}

# Example usage:
# datasets = load_dataset_with_splits('qnli')
# train_data, validation_data, test_data = datasets['train'], datasets['validation'], datasets['test']
