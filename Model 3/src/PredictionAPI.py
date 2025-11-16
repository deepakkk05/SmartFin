# save_model.py (Modern version)
import pickle
from pathlib import Path
from credit_score_model import *

# Create output directories using pathlib
Path('outputs/models').mkdir(parents=True, exist_ok=True)
print("✓ Directory created: outputs/models/")

# Train model
print("\nTraining model...")
results = main('cs-training.csv', 'cs-test.csv')

# Save components
print("\nSaving models...")

model_dir = Path('outputs/models')

(model_dir / 'model.pkl').write_bytes(pickle.dumps(results['model']))
print("✓ Model saved")

(model_dir / 'preprocessor.pkl').write_bytes(pickle.dumps(results['preprocessor']))
print("✓ Preprocessor saved")

(model_dir / 'shap_explainer.pkl').write_bytes(pickle.dumps(results['shap_explainer']))
print("✓ SHAP explainer saved")

(model_dir / 'what_if_analyzer.pkl').write_bytes(pickle.dumps(results['what_if_analyzer']))
print("✓ What-if analyzer saved")

print("\n✓ ALL MODELS SAVED SUCCESSFULLY!")
