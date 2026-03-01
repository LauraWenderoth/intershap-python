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

        self.model = self._auto_wrap_model(model)
        self.fusion = fusion
        self.transforms = transforms
        self.X_train = self._check_and_convert_X_data(X_train)
        self.n_modalities = len(self.X_train)
        # Set feature names if not provided
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            # Default: use string indices
            self.feature_names = [str(i) for i in range(self.n_modalities)]
        self._compute_mean_masks()
        self.train_dataset = IntershapDataset(
            self.X_train,
            labels=[0] * len(self.X_train[0]),
            fusion=fusion,
            transforms=transforms,
        )
        self.train_dataset.mask_mod = self.modality_masks
        self.base_value = self._calculate_base_value()
        self.coalitions = {}

    def _auto_wrap_model(self, model, class_index=None):
        if isinstance(model, BaseEstimator):
            return SklearnModelWrapper(model, class_index=class_index)
        return model

    def _compute_mean_masks(self):
        """
        Compute mean per modality over training data.
        Assumes modality can be converted to tensor.
        """
        mask_mod = []
        for i in range(self.n_modalities):
            values = []
            for j in range(len(self.X_train[i])):
                raw = self.X_train[i][j]
                # If transforms are provided, apply them
                if self.transforms and len(self.transforms) > i and self.transforms[i]:
                    loaded = self.transforms[i](raw)
                else:
                    loaded = raw
                if isinstance(loaded, torch.Tensor):
                    values.append(loaded)
                else:
                    try:
                        values.append(torch.tensor(loaded))
                    except Exception:
                        continue
            if len(values) > 0:
                stacked = torch.stack(values)
                mean_val = stacked.mean(dim=0)
                mask_mod.append(mean_val)
            else:
                mask_mod.append(None)
        self.modality_masks = mask_mod

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

    class MaskedDataset(Dataset):
        def __init__(self, base_dataset, mask):
            self.base_dataset = base_dataset
            self.mask = mask

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            x, y = self.base_dataset.__getitem__(idx, mask=self.mask)
            return x, y

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
        # Exclude the all-False (0000...) mask
        combos = list(product([True, False], repeat=self.n_modalities))
        return [c for c in combos if any(c)]

    def _calculate_base_value(self):
        # Use IntershapDataset to fuse mean masks, avoiding fusion logic duplication
        masked_dataset = self.MaskedDataset(
            self.train_dataset, mask=[False] * self.n_modalities
        )
        fused_sample, _ = masked_dataset[0]
        base_value = self.model(fused_sample)
        base_value = base_value.detach().cpu()
        if base_value.dim() == 0:
            base_value = base_value.unsqueeze(0).unsqueeze(1)
        elif base_value.dim() == 1:
            base_value = base_value.unsqueeze(1)
        return base_value

    def explain(self, X_test, y_test=None):
        X_test = self._check_and_convert_X_data(X_test)
        test_dataset = self._make_test_dataset(X_test, y_test)
        results = {}
        for mask in self.get_all_combinations():
            masked_dataset = self.MaskedDataset(test_dataset, mask)
            outputs = []
            for i in range(len(masked_dataset)):
                x, _ = masked_dataset[i]
                out = self.model(x)
                outputs.append(out.detach().cpu())
            results[mask] = torch.stack(outputs, dim=0)

        # Repeat base_value for each sample in test_dataset
        # Ensure base_value has shape [1, 1] (or [1, n_classes]) before repeat

        results[tuple([False] * self.n_modalities)] = self.base_value.repeat(
            len(test_dataset), *([1] * (self.base_value.dim() - 1))
        )
        self.coalitions = results
        self.shaply_values = self.calc_shaply_values()
        self.interaction_values = None
        return self.interaction_values

    def __call__(self, X_test, y_test=None):
        return self.explain(X_test, y_test)

    def calc_shaply_values(self):
        """
        Calculate Shapley values for each modality using:
        phi_i = sum_{S subset N \ {i}} [|S|! * (M-|S|-1)! / M!] * (f(S ∪ {i}) - f(S))
        Returns: dict mapping modality index to Shapley value tensor (shape: [n_samples, ...])
        """
        import itertools
        import math

        n = self.n_modalities
        M = n
        shap_values = {}
        modalities = list(range(n))
        # Get output shape from any coalition
        sample_out = next(iter(self.coalitions.values()))
        out_shape = sample_out.shape
        # For each modality i
        for i in modalities:
            phi_i = torch.zeros(out_shape)
            others = [m for m in modalities if m != i]
            # For all subsets S of N \ {i}
            for k in range(0, len(others) + 1):
                for S in itertools.combinations(others, k):
                    S_set = set(S)
                    # Build mask for S
                    mask_S = [m in S_set for m in modalities]
                    # Build mask for S ∪ {i}
                    mask_Si = mask_S.copy()
                    mask_Si[i] = True
                    # Factorial terms
                    fact_S = math.factorial(len(S))
                    fact_rest = math.factorial(M - len(S) - 1)
                    denom = math.factorial(M)
                    coeff = fact_S * fact_rest / denom
                    # Evaluate f(S ∪ {i}) and f(S)
                    out_Si = self.coalitions[tuple(mask_Si)]
                    out_S = self.coalitions[tuple(mask_S)]
                    phi_i += coeff * (out_Si - out_S)
            shap_values[i] = phi_i
        return shap_values
