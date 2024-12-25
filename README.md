
# B-Tagging with RetNet, XGB, LTC, and MLP

This repository provides tools and scripts for implementing and comparing various machine learning models—RetNet, XGBoost (XGB), Long Short-Term Memory (LSTM), and Multi-Layer Perceptron (MLP)—for b-jet tagging in high-energy physics experiments.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

B-tagging is a technique used in particle physics to identify jets originating from bottom quarks (b-quarks). Accurate b-tagging is crucial for analyses involving processes like Higgs boson decays and top quark studies. This repository explores the implementation of several machine learning models to enhance b-jet tagging performance.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/asugu/B-Tagging.git
   cd B-Tagging
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Ensure that your dataset is in the appropriate format for training and evaluation. The `jet_processor.py` script provides functions to preprocess raw data into a suitable format for model training. Additionally, `concat_pickle.py` can be used to concatenate multiple pickle files containing processed jet data.

## Model Training

The repository includes scripts for training different models:

- **XGBoost:** Use `XGB_train.py` to train an XGBoost model.

- **RetNet, LSTM, MLP:** The `model_trainer.py` script facilitates training these models. Model architectures are defined in `models.py`.

Configure hyperparameters and training settings within the respective scripts before execution.

## Evaluation

After training, evaluate model performance using the `histograms.ipynb` Jupyter Notebook, which provides tools to visualize metrics such as accuracy, precision, recall, and ROC curves.

## Usage

1. **Preprocess the data:**

   ```bash
   python jet_processor.py --input data/raw_data.csv --output data/processed_data.pkl
   ```

2. **Train a model (e.g., XGBoost):**

   ```bash
   python XGB_train.py --data data/processed_data.pkl --model_output models/xgb_model.pkl
   ```

3. **Evaluate the model:**

   Open `histograms.ipynb` in Jupyter Notebook and follow the instructions to load the trained model and visualize performance metrics.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your enhancements. Ensure that your code adheres to the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information on b-tagging and its significance in particle physics, refer to the [B-tagging Wikipedia page](https://en.wikipedia.org/wiki/B-tagging).
