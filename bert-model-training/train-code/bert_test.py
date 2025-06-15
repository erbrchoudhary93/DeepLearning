import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import json
import os

# Set up paths
BASE_DIR = "/Users/ram/Desktop/AI-Team-Work/ram-work/bert-training/train-code"
# MODEL_PATH = os.path.join(BASE_DIR, "model/finetuned/checkpoint-435")
MODEL_PATH = os.path.join(BASE_DIR, "model/finetuned/final_model")
DATA_PATH = os.path.join(BASE_DIR, "test_data_with_id.json")

class BERTPredictor:
    def __init__(self, model_path, original_model_name='emilyalsentzer/Bio_ClinicalBERT'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(original_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder
        # We'll need to initialize it with the same categories as during training
        self.le = self._initialize_label_encoder()
    
    def _initialize_label_encoder(self):
        # Load your training data to get the original categories
        df = pd.read_json(DATA_PATH)
        le = LabelEncoder()
        le.fit(df['category_id'])
        return le
    
    def predict(self, text):
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.softmax(dim=-1)
            predicted_class = predictions.argmax(dim=-1)
        
        # Convert numerical prediction back to category label
        predicted_category = self.le.inverse_transform(predicted_class.cpu().numpy())
        
        # Get confidence scores
        confidence_scores = predictions.cpu().numpy()[0]
        
        # Create results dictionary
        results = {
            'predicted_category': predicted_category[0],
            'confidence': float(confidence_scores[predicted_class[0]]),
            'all_probabilities': {
                category: float(prob)
                for category, prob in zip(self.le.classes_, confidence_scores)
            }
        }
        
        return results

def main():
    # Initialize predictor
    predictor = BERTPredictor(MODEL_PATH)
    
    # Test examples
    test_texts = [
        "The patient presented with severe chest pain and shortness of breath",
        "Blood pressure was elevated at 140/90",
        "Patient reports feeling anxious and having trouble sleeping"
    ]
    
    # Make predictions
    print("\nMaking predictions:")
    print("-" * 50)
    for text in test_texts:
        results = predictor.predict(text)
        print(f"\nInput text: {text}")
        print(f"Predicted category: {results['predicted_category']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print("\nAll category probabilities:")
        sorted_probs = sorted(results['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for category, prob in sorted_probs:
            print(f"{category}: {prob:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    main() 