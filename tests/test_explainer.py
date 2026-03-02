import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np
import torch

from intershap.explainer import Explainer


# Dummy model for testing (mimics sandbox.py logic)
class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


def test_shaply_additivity():
    """
    Test that the sum of base value and Shapley values for each modality equals the output for all modalities present (additivity).
    """
    N = 200
    d = 4
    std = 0.8
    vec1 = np.random.randint(0, 2, N)
    vec2 = np.random.randint(0, 2, N)
    label = vec1 | vec2
    modality1 = np.random.normal(loc=vec1[:, None], scale=std, size=(N, d))
    modality2 = np.random.normal(loc=vec2[:, None], scale=std, size=(N, d))
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = (
        modality1[:100],
        modality1[100:],
        modality2[:100],
        modality2[100:],
        label[:100],
        label[100:],
    )
    X_train = [X1_train, X2_train]
    X_test = [X1_test, X2_test]

    def transform(x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().float()
        return torch.tensor(x, dtype=torch.float32)

    model = LogisticRegressionTorch(X1_train.shape[1] + X2_train.shape[1])
    explainer = Explainer(
        model, X_train, fusion="flat", transforms=[transform, transform]
    )
    # Prepare test data
    X_test_torch = [
        torch.tensor(X1_test, dtype=torch.float32),
        torch.tensor(X2_test, dtype=torch.float32),
    ]
    explainer.explain(X_test_torch)
    shap_values = explainer.calc_shapley_values()
    base_mask = tuple([False, False])
    both_mask = tuple([True, True])
    base = explainer.coalitions[base_mask]
    mod0 = shap_values[0]
    mod1 = shap_values[1]
    output = explainer.coalitions[both_mask]
    # Test additivity for each row
    assert torch.allclose(base + mod0 + mod1, output, atol=1e-4), (
        " ❌ Shapley additivity test failed!"
    )


def test_intershap_decomposition():
    """
    Test that the sum of base value, all main effects (AA), and all pairwise interactions (AB) equals the coalition value for all modalities present (InterShap decomposition).
    """
    N = 200
    d = 4
    std = 0.8
    vec1 = np.random.randint(0, 2, N)
    vec2 = np.random.randint(0, 2, N)
    label = vec1 | vec2
    modality1 = np.random.normal(loc=vec1[:, None], scale=std, size=(N, d))
    modality2 = np.random.normal(loc=vec2[:, None], scale=std, size=(N, d))
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = (
        modality1[:100],
        modality1[100:],
        modality2[:100],
        modality2[100:],
        label[:100],
        label[100:],
    )
    X_train = [X1_train, X2_train]
    X_test = [X1_test, X2_test]

    def transform(x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().float()
        return torch.tensor(x, dtype=torch.float32)

    model = LogisticRegressionTorch(X1_train.shape[1] + X2_train.shape[1])
    explainer = Explainer(
        model, X_train, fusion="flat", transforms=[transform, transform]
    )
    X_test_torch = [
        torch.tensor(X1_test, dtype=torch.float32),
        torch.tensor(X2_test, dtype=torch.float32),
    ]
    explainer.explain(X_test_torch)
    n_mod = explainer.n_modalities
    base_mask = tuple([False] * n_mod)
    all_mask = tuple([True] * n_mod)
    base_value = explainer.coalitions[base_mask]
    output = explainer.coalitions[all_mask]
    interaction_values = explainer.interaction_values
    # Sum all AA and ABs
    total = base_value.clone()
    # Add all main effects (AA)
    for i in range(n_mod):
        total += interaction_values[i]
    # Add all pairwise interactions (AB)
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            total += interaction_values[(i, j)]
    # Check decomposition
    assert torch.allclose(total, output, atol=1e-4), (
        " ❌ InterShap decomposition test failed!"
    )


if __name__ == "__main__":
    test_shaply_additivity()
    print("✅ Shapley additivity test passed.")
    test_intershap_decomposition()
    print("✅ InterShap decomposition test passed.")
