from itertools import product

import torch
from torch.utils.data import Dataset

from intershap.dataset import IntershapDataset


class Explainer:
    def __init__(
        self, model, X_train, feature_names=None, fusion="flat", transforms=None
    ):
        """
        model: trained model (callable)
        X_train: list of arrays (modalities) or array (single modality)
        feature_names: list of feature names (optional)
        fusion: fusion method for IntershapDataset
        transforms: list of transforms for each modality (optional)
        """
        if not isinstance(X_train, list):
            X_train = [X_train]
        self.model = model
        self.feature_names = feature_names
        self.fusion = fusion
        self.transforms = transforms
        self.train_dataset = IntershapDataset(
            X_train, labels=[0] * len(X_train[0]), fusion=fusion, transforms=transforms
        )
        self.n_modalities = len(X_train)
        self.results = None

    def _make_dataset(self, X, y=None):
        if not isinstance(X, list):
            X = [X]
        if y is None:
            y = [0] * len(X[0])
        return IntershapDataset(X, y, fusion=self.fusion, transforms=self.transforms)

    def get_all_combinations(self):
        return list(product([True, False], repeat=self.n_modalities))

    def explain(self, X_test, y_test=None):
        test_dataset = self._make_dataset(X_test, y_test)
        results = {}
        for mask in self.get_all_combinations():

            class MaskedDataset(Dataset):
                def __init__(self, base_dataset, mask):
                    self.base_dataset = base_dataset
                    self.mask = mask

                def __len__(self):
                    return len(self.base_dataset)

                def __getitem__(self, idx):
                    x, y = self.base_dataset.__getitem__(idx, mask=self.mask)
                    return x, y

            masked_dataset = MaskedDataset(test_dataset, mask)
            outputs = []
            for i in range(len(masked_dataset)):
                x, _ = masked_dataset[i]
                out = self.model(x)
                outputs.append(out.detach().cpu())
            results[mask] = torch.stack(outputs, dim=0)
        self.results = results
        return results

    def __call__(self, X_test, y_test=None):
        return self.explain(X_test, y_test)
