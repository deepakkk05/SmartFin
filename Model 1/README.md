# Expense Categorizer Model

A machine learning model for categorizing expenses using PyTorch and Transformers.

## Project Structure
- `install_dependencies.py`: Script to install required dependencies
- `category_model/`: Contains model checkpoints and training states
  - Only the best performing checkpoint (500) is included

## Setup
1. Clone the repository
2. Run the installation script:
```bash
python install_dependencies.py
```

## Model Performance
- Final accuracy: 87%
- Training epochs: 5
- Best checkpoint: 500 steps

## Dependencies
- PyTorch (CPU version)
- Transformers 4.30.0
- Accelerate 0.20.3
- Other dependencies listed in install_dependencies.py
