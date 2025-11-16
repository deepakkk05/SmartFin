from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib

# Paths
model_path = "./category_model/checkpoint-500"
base_model = "distilbert-base-uncased"
label_encoder_path = "./category_model/label_encoder.pkl"  # update path

# Load tokenizer, model, and label encoder
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
label_encoder = joblib.load(label_encoder_path)




# Prepare inputs without token_type_ids
text = "paid electricity bill"
inputs = tokenizer(text, return_tensors="pt")
if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

predicted_class_name = label_encoder.inverse_transform([predicted_class_id])[0]
print(f"Predicted category: {predicted_class_name}")
