from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from torch.utils.data import Dataset

from intershap.dataset import IntershapDataset


class SklearnModelWrapper:
    (
        """''
    Wraps a scikit-learn model to make it compatible with PyTorch-based explainers.
    This is needed because scikit-learn models expect numpy arrays as input, while
    the explainer expects a callable that takes a torch tensor and returns a torch tensor.
    The wrapper:
        - Converts torch tensor input to numpy array
        - Handles reshaping for single samples
        - Calls the appropriate scikit-learn method (decision_function or predict_proba)
        - Converts the output back to a torch tensor
    Without this wrapper, passing a torch tensor directly to a scikit-learn model would result in an error.
    """
        ""
    )

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

        # Convert each modality in X_train to torch.tensor if it's a numpy array or pandas DataFrame/Series and no transform is provided for that modality
        self.model = self._auto_wrap_model(model)
        self.feature_names = feature_names
        self.fusion = fusion
        self.transforms = transforms

        self.X_train = self._check_and_convert_X_data(X_train)
        self.train_dataset = IntershapDataset(
            self.X_train,
            labels=[0] * len(self.X_train[0]),
            fusion=fusion,
            transforms=transforms,
        )
        self.n_modalities = len(self.X_train)
        self.results = None

    def _check_and_convert_X_data(self, X):
        if not isinstance(X, list):
            raise ValueError(
                "Only one modality given. Please provide a list with modalities."
            )
        for i, x in enumerate(X):
            needs_transform = self.transforms is None or (
                self.transforms[i] is None
                if self.transforms and len(self.transforms) > i
                else True
            )
            if needs_transform:
                if isinstance(x, np.ndarray):
                    X[i] = torch.tensor(x)
                elif isinstance(x, (pd.DataFrame, pd.Series)):
                    X[i] = torch.tensor(x.values)
        return X

    def _make_test_dataset(self, X, y=None):
        if not isinstance(X, list):
            raise ValueError(
                "Only one modality given. Please provide a list with modalities."
            )
        if y is None:
            y = [0] * len(X[0])
        dataset = IntershapDataset(
            X,
            y,
            fusion=self.fusion,
            transforms=self.transforms,
            compute_mean_mask=False,
        )
        dataset.mask_mod = self.train_dataset.mask_mod
        return dataset

    def get_all_combinations(self):
        return list(product([True, False], repeat=self.n_modalities))

    def explain(self, X_test, y_test=None):
        X_test = self._check_and_convert_X_data(X_test)
        test_dataset = self._make_test_dataset(X_test, y_test)
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

    def _auto_wrap_model(self, model, class_index=None):
        if isinstance(model, BaseEstimator):
            return SklearnModelWrapper(model, class_index=class_index)
        return model
