import torch

from src.metrics.base_metric import BaseMetric


class CodebookPerplexityMetric(BaseMetric):
    def __init__(self, codebook_size, *args, **kwargs):
        super().__init__(name="codebook_perplexity", *args, **kwargs)
        self.codebook_size = codebook_size

    def __call__(self, indices, **batch):
        perplexities = []

        num_quantizers = indices.shape[-1]

        for q in range(num_quantizers):
            q_indices = indices[:, :, q].reshape(-1)

            counts = torch.bincount(q_indices, minlength=self.codebook_size)
            probs = counts / counts.sum()
            nonzero_probs = probs[probs > 0]
            entropy = -(nonzero_probs * torch.log(nonzero_probs)).sum()
            perplexity = torch.exp(entropy)
            perplexities.append(perplexity)

        perplexities = torch.stack(perplexities)

        return perplexities.mean().item()
