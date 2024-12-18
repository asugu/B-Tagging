
from model_dataset import load_datasets, JetParticleDataset
import torch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
import pickle
import os
from datetime import datetime
from utils import plot_metrics
import time

key = 'merged'

train_file = f'data/train_dataset_{key}.pt'  
test_file = f'data/test_dataset_{key}.pt'    
xgb_model_file = f'model/xgb_model_{key}.pkl'
output_dir = "checkpoints"       
plot_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Function to load PyTorch dataset and prepare it for XGBoost
def load_and_prepare_data(pt_file, save_npz_path=None):
    """
    Load, prepare, and optionally save data as a single .npz file.

    Parameters:
    - pt_file: str, Path to the PyTorch dataset file.
    - save_npz_path: str or None, Path to save the features and labels as a .npz file. If None, data is not saved.

    Returns:
    - features: np.ndarray, Prepared feature matrix.
    - labels: np.ndarray, Corresponding labels array.
    """
    print(f"Loading dataset from {pt_file}...")
    data = torch.load(pt_file)  # Load PyTorch dataset
    combined_inputs, labels, flavors = zip(*data)  # Unpack inputs and labels

    # Extract particle, vertex, and jet features
    particle_inputs, vertex_inputs, jet_inputs = zip(*combined_inputs)

    # Flatten and concatenate all features into a single array for each sample
    features = np.array([
        torch.cat([particle.flatten(), vertex.flatten(), jet], dim=0).cpu().numpy()
        for particle, vertex, jet in zip(particle_inputs, vertex_inputs, jet_inputs)
    ])

    # Convert labels to a numpy array
    labels = np.array(torch.tensor(labels).cpu().numpy())
    flavors = np.array(torch.tensor(flavors).cpu().numpy())

    # Save features and labels together if a path is provided
    if save_npz_path:
        np.savez(save_npz_path, features=features, labels=labels, flavors=flavors)
        print(f"Features and labels saved together to {save_npz_path}")

    return features, labels, flavors

# Load training and testing data
# X_train, y_train, F_train = load_and_prepare_data(train_file, save_npz_path=f"data/train_dataset_{key}.npz")
# X_test, y_test, F_test = load_and_prepare_data(test_file, save_npz_path=f"data/test_dataset_{key}.npz")


# Load datasets from saved .npz files
train_data = np.load(f"data/train_dataset_{key}.npz")
test_data = np.load(f"data/test_dataset_{key}.npz")
X_train, y_train, F_train = train_data['features'], train_data['labels'], train_data['flavors']
X_test, y_test, F_test = test_data['features'], test_data['labels'], test_data['flavors']

jet_pt_index = -5 
train_jet_pt = X_train[:, jet_pt_index]
test_jet_pt = X_test[:, jet_pt_index]

print("Training XGBoost model...")

xgb_model = XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,            # Number of boosting rounds
    learning_rate=0.1,           # Step size shrinkage
    max_depth=13,                # Maximum depth of a tree
)

xgb_model.fit(X_train, y_train)

# Save the trained model
with open(os.path.join(output_dir, xgb_model_file), 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"Model saved to {xgb_model_file}")


print("Evaluating the model...")
begin_time = time.time()
y_pred = xgb_model.predict(X_test)
end_time = time.time()
print("Inference time: ",(end_time - begin_time), len(X_test))
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability for ROC-AUC

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred, average='weighted')

# Save evaluation data
val_data_dict = {
    'preds': y_pred.tolist(),
    'probs': y_pred_proba.tolist(),
    'labels': y_test.tolist(),
    'features': X_test.tolist(),
    'flavors': F_test.tolist(),
    'jet_pt': test_jet_pt.tolist()
}
train_data_dict = {
    'preds': xgb_model.predict(X_train).tolist(),
    'probs': xgb_model.predict_proba(X_train)[:, 1].tolist(),
    'labels': y_train.tolist(),
    'features': X_train.tolist(),
    'flavors': F_train.tolist(),
    'jet_pt': train_jet_pt.tolist()
}

with open(os.path.join(output_dir, f'val_data/val_data_dict_XGB.pkl'), 'wb') as f:
    pickle.dump(val_data_dict, f)
with open(os.path.join(output_dir, f'val_data/train_data_dict_XGB.pkl'), 'wb') as f:
    pickle.dump(train_data_dict, f)

# Save metrics
metrics = {
    'accuracy': accuracy,
    'auc': auc,
    'f1_score': f1,
    'train_size': len(y_train),
    'test_size': len(y_test),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
with open(os.path.join(output_dir, f'metrics_XGB.pkl'), 'wb') as f:
    pickle.dump(metrics, f)

print(f"Metrics saved:\nAccuracy: {accuracy:.4f}\nAUC: {auc:.4f}\nF1 Score: {f1:.4f}")

# Save final checkpoint
checkpoint = {
    'model': xgb_model,
    'train_data_dict': train_data_dict,
    'val_data_dict': val_data_dict,
    'metrics': metrics
}
checkpoint_path = os.path.join(output_dir, f'model/final_checkpoint_XGB.pkl')
with open(checkpoint_path, 'wb') as f:
    pickle.dump(checkpoint, f)
print(f"Final checkpoint saved to {checkpoint_path}")
