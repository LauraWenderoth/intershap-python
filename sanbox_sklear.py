# %%
import sys

sys.path.append("/Users/lwenderoth/Documents/intershap-python/src")
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from intershap.explainer import Explainer

# Parameters
N = 10000  # number of entries
d = 10  # dimension of each vector
std = 0.8

# Randomly sample two binary vectors of length N
vec1 = np.random.randint(0, 2, N)
vec2 = np.random.randint(0, 2, N)
label = vec1 | vec2

# For each entry, generate two d-dimensional vectors
# The mean of each vector is either 0 or 1 depending on the sampled value
modality1 = np.random.normal(loc=vec1[:, None], scale=std, size=(N, d))
modality2 = np.random.normal(loc=vec2[:, None], scale=std, size=(N, d))

# Split modalities and labels
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    modality1, modality2, label, test_size=0.2, random_state=42
)

# Store modalities as a list for training and testing
X_train = [X1_train, X2_train]
X_test = [X1_test, X2_test]

# Prepare flat features for sklearn
X_train_flat = np.concatenate([X1_train, X2_train], axis=1)
X_test_flat = np.concatenate([X1_test, X2_test], axis=1)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X_train_flat, y_train)

# Predict on test set
y_pred = clf.predict(X_test_flat)

# Evaluate metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Define a model wrapper for sklearn (decision_function for SHAP-like output)
class SklearnModelWrapper:
    def __init__(self, model, class_index=None):
        self.model = model
        self.class_index = class_index  # If None, return all classes

    def __call__(self, X):
        X_np = X.numpy()
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_np)
            if scores.ndim == 1:
                # Binary classification: scores is 1D
                if self.class_index is None:
                    return torch.tensor(scores)
                elif self.class_index == 0:
                    return torch.tensor(scores)
                elif self.class_index == 1:
                    return torch.tensor(-scores)
                else:
                    raise IndexError(
                        "class_index out of range for binary classification"
                    )
            else:
                # Multiclass: scores is 2D
                if self.class_index is not None:
                    return torch.tensor(scores[:, self.class_index])
                else:
                    return torch.tensor(scores)
        else:
            # Fallback to predict_proba if decision_function not available
            probs = self.model.predict_proba(X_np)
            if self.class_index is not None:
                return torch.tensor(probs[:, self.class_index])
            else:
                return torch.tensor(probs)

# Example: explain for class 0 using new API
def transform(x):
    return torch.tensor(x, dtype=torch.float32)
feature_names = None  # or provide names if available
explainer = Explainer(
    SklearnModelWrapper(clf, class_index=0),
    [X1_train, X2_train],
    feature_names=feature_names,
    fusion="flat",
    transforms=[transform, transform]
)
shap_values = explainer([X1_test, X2_test])

if explainer.results is not None:
    for mask, output in explainer.results.items():
        print(f"Mask: {mask}, Output shape: {output.shape}")


results = explainer.results.items()
# Convert results to DataFrame with mask columns and output values as rows
results_dict = {str(mask): output.flatten().tolist() for mask, output in results}
df_results = pd.DataFrame(results_dict)
# %%

# %%

# %%

# %%

# %%

# %%
