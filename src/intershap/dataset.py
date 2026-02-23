import torch
from torch.utils.data import Dataset


class IntershapDataset(Dataset):
    def __init__(
        self,
        modalities,
        labels,
        loaders=None,
        transforms=None,
        fusion=None,
        compute_mean_mask=True,
        feature_names=None,
    ):
        """
        modalities: list of lists (one list per modality)
                    Each modality must have same length.
        labels: list or tensor
        loaders: list of callables per modality (optional)
        transforms: list of callables per modality (optional)
        fusion: "flat", "channel", or callable for fusing modalities (optional)
            - "flat": flatten each modality and concatenate
            - "channel": concatenate along channel dimension (for images)
            - callable: custom fusion function that takes list of samples and returns fused sample
        """

        self.modalities = modalities
        self.labels = labels
        self.n_modalities = len(modalities)
        self.feature_names = feature_names

        if loaders is not None and len(loaders) != self.n_modalities:
            raise ValueError("Length of loaders must match number of modalities")
        self.loaders = loaders or [None] * self.n_modalities

        if transforms is not None and len(transforms) != self.n_modalities:
            raise ValueError("Length of transforms must match number of modalities")
        self.transforms = transforms or [None] * self.n_modalities
        self.fusion = fusion
        assert all(len(m) == len(labels) for m in modalities), (
            "All modalities must have same number of samples"
        )

        self.mask_mod = []

        if compute_mean_mask:
            self._compute_mean_masks()

    def __len__(self):
        return len(self.labels)

    def _load_modality(self, modality_idx, sample):
        loader = self.loaders[modality_idx]
        transform = self.transforms[modality_idx]

        if loader:
            sample = loader(sample)

        if transform:
            sample = transform(sample)

        return sample

    def _compute_mean_masks(self):
        """
        Compute mean per modality over dataset.
        Assumes modality can be converted to tensor.
        """
        for i in range(self.n_modalities):
            values = []

            for j in range(len(self)):
                raw = self.modalities[i][j]
                loaded = self._load_modality(i, raw)

                if isinstance(loaded, torch.Tensor):
                    values.append(loaded)
                else:
                    try:
                        values.append(torch.tensor(loaded))
                    except:
                        continue

            if len(values) > 0:
                stacked = torch.stack(values)
                mean_val = stacked.mean(dim=0)
                self.mask_mod.append(mean_val)
            else:
                self.mask_mod.append(None)

    def __getitem__(self, idx, mask=None):
        """
        mask: list/tuple of bools (len = n_modalities), True=keep, False=mask (replace with mean)
        If mask is None, no masking is applied.
        """
        samples = []
        for i in range(self.n_modalities):
            raw = self.modalities[i][idx]
            sample = self._load_modality(i, raw)
            if mask is not None and not mask[i]:
                # Replace with mean mask if available
                if self.mask_mod[i] is not None:
                    sample = self.mask_mod[i]
            samples.append(sample)
        label = self.labels[idx]
        if self.fusion:
            samples = self._apply_fusion(samples, self.fusion)
        return samples, label

    def _apply_fusion(self, samples, fusion):
        if fusion == "flat":
            return self._flat_fusion(samples)

        elif fusion == "channel":
            return self._channel_fusion(samples)

        elif callable(fusion):
            return fusion(samples)

        else:
            raise ValueError("Unknown fusion type")

    def _flat_fusion(self, samples):
        flat = []
        for s in samples:
            if not isinstance(s, torch.Tensor):
                raise ValueError("All modalities must be tensors")
            flat.append(s.flatten())
        return torch.cat(flat)

    def _channel_fusion(self, samples):
        for s in samples:
            if not isinstance(s, torch.Tensor):
                raise ValueError("Channel fusion requires tensors")

            if s.dim() != 3:
                raise ValueError("Expected tensors of shape (C,H,W)")

        spatial = [s.shape[-2:] for s in samples]

        if len(set(spatial)) != 1:
            raise ValueError("All images must have same H,W")

        return torch.cat(samples, dim=0)

    def _concatenate(self, samples):
        """
        Concatenate only if tensors and 1D compatible.
        """
        flat = []
        for s in samples:
            if not isinstance(s, torch.Tensor):
                raise ValueError("All modalities must be tensors for concatenation")

            if s.dim() > 1:
                raise ValueError("Cannot concatenate multi-dimensional tensors")

            flat.append(s)

        return torch.cat(flat)
