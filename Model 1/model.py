import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import re
import warnings
import os
from typing import Optional, Dict, List
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ExpenseCategorizer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.category_model = None
        self.subcategory_model = None
        self.category_encoder = LabelEncoder()
        self.subcategory_encoder = LabelEncoder()
        self.max_length = 128
        self.ner_pipeline = None
        self.nlp = None
        self._initialize_nlp_components()

    def _initialize_nlp_components(self):
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=-1,
                return_all_scores=False
            )
            logger.info("NER pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load NER pipeline: {e}")
            self.ner_pipeline = None

        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Attempting to download...")
                import subprocess
                import sys
                subprocess.run([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ], check=True, capture_output=True)
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model downloaded and loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load spaCy: {e}")
            self.nlp = None

    def preprocess_text(self, text: str) -> str:
        if pd.isna(text) or text is None:
            return ""
        try:
            text = str(text).lower().strip()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            if len(text.strip()) < 2:
                return text
            entity_info = []
            if self.ner_pipeline:
                try:
                    entities = self.ner_pipeline(text)
                    for entity in entities:
                        if entity.get('entity_group') in ['ORG', 'PER', 'LOC']:
                            entity_word = entity.get('word', '').replace('##', '')
                            entity_info.append(f"{entity['entity_group'].lower()}_{entity_word.lower()}")
                except Exception:
                    pass
            locations, organizations = [], []
            if self.nlp:
                try:
                    doc = self.nlp(text)
                    locations = [ent.text.lower() for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
                    organizations = [ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"]
                except Exception:
                    pass
            enhanced_text = text
            if entity_info:
                enhanced_text += " " + " ".join(entity_info)
            if locations:
                enhanced_text += " " + " ".join([f"loc_{loc}" for loc in locations])
            if organizations:
                enhanced_text += " " + " ".join([f"org_{org}" for org in organizations])
            return enhanced_text.strip()
        except Exception:
            return str(text).lower().strip()

    def create_keyword_features(self, descriptions: pd.Series) -> pd.DataFrame:
        category_keywords = {
            'utilities': ['electricity', 'water', 'gas', 'internet', 'wifi', 'bill', 'utility', 'phone'],
            'food': ['restaurant', 'cafe', 'food', 'lunch', 'dinner', 'grocery', 'snack', 'pizza', 'coffee'],
            'transport': ['taxi', 'bus', 'metro', 'fuel', 'petrol', 'travel', 'flight', 'uber', 'train'],
            'healthcare': ['doctor', 'medical', 'hospital', 'medicine', 'dental', 'health', 'pharmacy'],
            'education': ['school', 'college', 'book', 'course', 'fees', 'tuition', 'university'],
            'shopping': ['buy', 'purchase', 'shop', 'clothes', 'shoes', 'electronics', 'amazon'],
            'entertainment': ['movie', 'game', 'netflix', 'spotify', 'event', 'concert', 'theater'],
            'insurance': ['insurance', 'premium', 'policy', 'claim'],
            'housing': ['rent', 'mortgage', 'repair', 'furniture', 'home', 'apartment'],
            'taxes': ['tax', 'government', 'service tax', 'income tax', 'irs']
        }
        keyword_features = []
        for desc in descriptions:
            desc_lower = str(desc).lower() if pd.notna(desc) else ""
            features = {}
            for cat, keywords in category_keywords.items():
                features[f'has_{cat}'] = any(keyword in desc_lower for keyword in keywords)
            keyword_features.append(features)
        return pd.DataFrame(keyword_features)


class ExpenseDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}


class HybridExpenseClassifier:
    def __init__(self):
        self.categorizer = ExpenseCategorizer()
        self.category_model = None

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['DESCRIPTION', 'CATEGORY', 'SUBCATEGORY']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        initial_len = len(df)
        df = df.dropna(subset=['DESCRIPTION', 'CATEGORY'])
        if len(df) == 0:
            raise ValueError("No valid data rows after removing missing values")
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} rows with missing values")
        df['SUBCATEGORY'] = df['SUBCATEGORY'].fillna('Unknown')
        return df.reset_index(drop=True)

    def analyze_data_distribution(self, df: pd.DataFrame):
        logger.info("=== Data Distribution Analysis ===")
        cat_counts = df['CATEGORY'].value_counts()
        for cat, count in cat_counts.head(10).items():
            logger.info(f"  {cat}: {count} samples")
        rare_cats = cat_counts[cat_counts < 3]
        if len(rare_cats) > 0:
            logger.warning(f"Categories with < 3 samples: {len(rare_cats)}")
            for cat, count in rare_cats.items():
                logger.warning(f"  {cat}: {count} samples")
        return cat_counts

    def load_and_preprocess_data(self, df: pd.DataFrame):
        df = self.validate_dataframe(df)
        logger.info(f"Processing {len(df)} expense records...")
        self.analyze_data_distribution(df)
        df['processed_description'] = df['DESCRIPTION'].apply(self.categorizer.preprocess_text)
        keyword_features = self.categorizer.create_keyword_features(df['DESCRIPTION'])
        df['category_encoded'] = self.categorizer.category_encoder.fit_transform(df['CATEGORY'])
        df['subcategory_encoded'] = self.categorizer.subcategory_encoder.fit_transform(df['SUBCATEGORY'])
        logger.info(f"Found {len(df['CATEGORY'].unique())} unique categories")
        logger.info(f"Found {len(df['SUBCATEGORY'].unique())} unique subcategories")
        return df, keyword_features

    def consolidate_rare_categories(self, df: pd.DataFrame, min_samples: int = 3) -> pd.DataFrame:
        cat_counts = df['CATEGORY'].value_counts()
        rare_categories = cat_counts[cat_counts < min_samples].index.tolist()
        if len(rare_categories) > 0:
            logger.info(f"Consolidating {len(rare_categories)} rare categories into 'Other'")
            df.loc[df['CATEGORY'].isin(rare_categories), 'CATEGORY'] = 'Other'
            df.loc[df['CATEGORY'] == 'Other', 'SUBCATEGORY'] = 'Other'
        return df

    def train_category_model(self, df: pd.DataFrame):
        if len(df) < 10:
            raise ValueError("Not enough data to train model (minimum 10 samples required)")
        
        class_counts = df['category_encoded'].value_counts()
        if class_counts.min() < 2:
            df = self.consolidate_rare_categories(df)
            self.categorizer.category_encoder = LabelEncoder()
            df['category_encoded'] = self.categorizer.category_encoder.fit_transform(df['CATEGORY'])
        
        can_stratify = df['category_encoded'].value_counts().min() >= 2 and len(df['category_encoded'].unique()) > 1
        split_params = {'test_size': 0.2, 'random_state': 42}
        if can_stratify:
            split_params['stratify'] = df['category_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_description'], df['category_encoded'], **split_params
        )
        
        train_dataset = ExpenseDataset(
            X_train.tolist(), 
            y_train.tolist(), 
            self.categorizer.tokenizer, 
            self.categorizer.max_length
        )
        test_dataset = ExpenseDataset(
            X_test.tolist(), 
            y_test.tolist(), 
            self.categorizer.tokenizer, 
            self.categorizer.max_length
        )
        
        num_categories = len(df['CATEGORY'].unique())
        
        # Initialize model and explicitly move to CPU
        self.category_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=num_categories
        )
        
        # Explicitly set device
        device = torch.device('cpu')
        self.category_model = self.category_model.to(device)
        
        training_args = TrainingArguments(
            output_dir='./category_model',
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to=None,
            no_cuda=True,
            use_mps_device=False
        )
        
        trainer = Trainer(
            model=self.category_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Category Classification Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        labels_list = list(range(len(self.categorizer.category_encoder.classes_)))
        print(classification_report(
            y_test,
            y_pred,
            labels=labels_list,
            target_names=self.categorizer.category_encoder.classes_,
            zero_division=0
        ))
        
        return accuracy, y_test, y_pred


def train_model():
    """Main training function"""
    try:
        dataset_path = 'cleaned_expenses_dataset.csv'
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file '{dataset_path}' not found!")
            return None, None, None
        
        logger.info(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        clf = HybridExpenseClassifier()
        df, keyword_features = clf.load_and_preprocess_data(df)
        accuracy, y_test, y_pred = clf.train_category_model(df)
        
        # Save label encoders
        os.makedirs('./category_model', exist_ok=True)
        joblib.dump(clf.categorizer.category_encoder, './category_model/label_encoder.pkl')
        joblib.dump(clf.categorizer.subcategory_encoder, './category_model/subcategory_encoder.pkl')
        logger.info("Label encoders saved successfully")
        
        if len(df) > 0:
            sample_description = df['DESCRIPTION'].iloc[0]
            logger.info(f"Test Description: '{sample_description}'")
        
        return clf, y_test, y_pred
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise


def visualize_confusion_matrix(clf, y_test, y_pred):
    """Create and display confusion matrix"""
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=clf.categorizer.category_encoder.classes_
        )
        disp.plot(cmap='Blues', xticks_rotation=90)
        plt.title("Confusion Matrix - Expense Category Classification")
        plt.tight_layout()
        plt.savefig('./category_model/confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info("Confusion matrix saved to './category_model/confusion_matrix.png'")
        plt.show()
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e}")


def inference(text: str, model_path: str = './category_model'):
    """Run inference on a single text"""
    try:
        # Load model and encoder
        base_model = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
        
        # Prepare inputs
        inputs = tokenizer(text, return_tensors="pt")
        
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
        
        predicted_class_name = label_encoder.inverse_transform([predicted_class_id])[0]
        logger.info(f"Text: '{text}' -> Predicted category: '{predicted_class_name}'")
        return predicted_class_name
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise


if __name__ == "__main__":
    # Training phase
    logger.info("=== Starting Training ===")
    clf, y_test, y_pred = train_model()
    
    if clf is not None and y_test is not None and y_pred is not None:
        # Visualization phase
        logger.info("\n=== Creating Confusion Matrix ===")
        visualize_confusion_matrix(clf, y_test, y_pred)
        
        # Inference phase
        logger.info("\n=== Running Sample Inference ===")
        test_texts = [
            "paid to shop",
            "electricity bill payment",
            "lunch at restaurant"
        ]
        
        for text in test_texts:
            try:
                predicted = inference(text)
                print(f"'{text}' -> {predicted}")
            except Exception as e:
                logger.error(f"Failed to predict for '{text}': {e}")
