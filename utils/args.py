import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='DataMap Construction.')

parser.add_argument(
    '--model-name',
    type=str,
    required=True,
    help='name of the model to use, e.g., bert-base-cased',
    choices=['distilbert-base-cased', 'bert-base-cased', 'bert-large-cased', 'distilroberta-base', 'roberta-base', 'roberta-large']
)

parser.add_argument(
    '--dataset-name',
    type=str,
    required=True,
    help='name of the dataset to use, e.g., qnli, advsquad',
    choices=['qnli', 'winogrande', 'advsquad', 'wsc']
)