# %%
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from intershap.dataset import IntershapDataset

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

# Create Dataset and DataLoader
train_dataset = IntershapDataset(modalities=X_train, labels=y_train)
test_dataset = IntershapDataset(modalities=X_test, labels=y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Logistic Regression model in PyTorch
class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


# Training loop
input_dim = X1_train.shape[1] + X2_train.shape[1]
model = LogisticRegressionTorch(input_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(5):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Test accuracy: {acc:.3f}")
print("Classification report:\n", classification_report(all_labels, all_preds))
# %%


dataset = IntershapDataset(modalities=[X1_train, X2_train], labels=y_train)

# %%
