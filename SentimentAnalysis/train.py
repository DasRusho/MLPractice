from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

print(torch.__version__)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

results = classifier(["We are very happy to show you the huggingface transformers library.",
                    "We hope you don't hate it."])
for result in results:
    print(result)

tokens = tokenizer.tokenize("We are very happy to show you the huggingface transformers library.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("We are very happy to show you the huggingface transformers library.")

print(f'Tokens : {tokens}')
print(f'Token IDs : {token_ids}')
print(f'Input IDs : {input_ids}')
