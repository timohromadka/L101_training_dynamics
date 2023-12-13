from transformers import AutoModel, AutoTokenizer

def load_model_and_tokenizer(model_name):
    """
    Loads a specified model and its tokenizer from Hugging Face's Transformers library.

    Args:
    model_name (str): The name of the model to load (e.g., 'bert-base-cased', 'roberta-base').

    Returns:
    model: The loaded model.
    tokenizer: The corresponding tokenizer for the model.
    """
    # Supported models
    supported_models = [
        'bert-base-cased', 'bert-large-cased',
        'roberta-base', 'roberta-large',
        'distilbert-base-cased', 'distilroberta-base'
    ]

    if model_name not in supported_models:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from {supported_models}")

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

# Example usage:
# model, tokenizer = load_model_and_tokenizer('bert-base-cased')
