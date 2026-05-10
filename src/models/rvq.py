import torch
from torch import nn


def dist(x, y):
    return (x**2).sum(dim=1, keepdim=True) - 2 * x @ y.T + (y**2).sum(dim=1)


class VectorQuantizer(nn.Module):
    def __init__(
        self, latent_channels, codebook_size, gamma, kmeans_n_iters, code_threshold
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.gamma = gamma
        self.latent_channels = latent_channels
        self.kmeans_n_iters = kmeans_n_iters
        self.code_threshold = code_threshold
        self.initialized = False

        self.register_buffer("codebook", torch.randn(codebook_size, latent_channels))
        self.register_buffer("ema_quantized_number", torch.zeros(codebook_size))
        self.register_buffer("ema_cum_sum", self.codebook.clone())

    def kmeans_init(self, residual, n_iters):
        n_vectors = residual.shape[0]
        random_indices = torch.randperm(n_vectors, device=residual.device)[
            : self.codebook_size
        ]
        centroids = residual[random_indices]

        for _ in range(n_iters):
            distances = dist(residual, centroids)

            indices = distances.argmin(dim=1)
            new_centroids = torch.zeros_like(centroids)
            new_centroids.index_add_(0, indices, residual)
            counts = torch.bincount(indices, minlength=self.codebook_size).to(
                residual.dtype
            )
            used = counts > 0

            new_centroids[used] = new_centroids[used] / counts[used].unsqueeze(1)
            new_centroids[~used] = centroids[~used]
            centroids = new_centroids

        self.codebook.copy_(centroids)
        self.ema_cum_sum.copy_(centroids * self.code_threshold)
        self.ema_quantized_number.fill_(self.code_threshold)

    def forward(self, residual, **batch):
        B, T = residual.shape[0], residual.shape[1]
        residual = residual.reshape(-1, self.latent_channels)

        if self.training and not self.initialized:
            with torch.no_grad():
                self.kmeans_init(residual, self.kmeans_n_iters)
            self.initialized = True

        cur_codebook = self.codebook
        distances = dist(residual, cur_codebook)

        indices = distances.argmin(dim=1)
        quantized = cur_codebook[indices]
        quantized = quantized.reshape(B, T, self.latent_channels)

        if self.training:
            with torch.no_grad():
                quantized_number = torch.bincount(indices, minlength=self.codebook_size)
                self.ema_quantized_number.mul_(self.gamma).add_(
                    quantized_number * (1 - self.gamma)
                )

                codebook_update = torch.zeros_like(cur_codebook)
                codebook_update.index_add_(0, indices, residual, alpha=1 - self.gamma)
                self.ema_cum_sum.mul_(self.gamma).add_(codebook_update)

                unused_codes = self.ema_quantized_number < self.code_threshold
                if unused_codes.any():
                    n_dead = unused_codes.sum().item()
                    random_indices = torch.randint(
                        low=0,
                        high=residual.shape[0],
                        size=(n_dead,),
                        device=residual.device,
                    )

                    new_vectors = residual[random_indices]
                    self.ema_quantized_number[unused_codes] = self.code_threshold
                    self.ema_cum_sum[unused_codes] = new_vectors * self.code_threshold

                new_codebook = self.ema_cum_sum / self.ema_quantized_number.unsqueeze(
                    1
                ).clamp_min(1e-8)
                self.codebook.copy_(new_codebook)

        residual = residual.reshape(B, T, self.latent_channels)
        commitment_loss = ((residual - quantized.detach()) ** 2).mean()

        return {
            "quantized": quantized,
            "indices": indices.reshape(B, T),
            "commitment_loss": commitment_loss,
        }


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_quantizers,
        latent_channels,
        codebook_size,
        gamma,
        kmeans_n_iters,
        code_threshold,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    latent_channels,
                    codebook_size,
                    gamma,
                    kmeans_n_iters,
                    code_threshold,
                )
                for _ in range(self.num_quantizers)
            ]
        )

    def forward(self, encoded):
        B, T = encoded.shape[0], encoded.shape[1]

        residual = encoded
        cum_quantized = torch.zeros_like(encoded)
        all_indices = torch.zeros(
            (B, T, self.num_quantizers), dtype=torch.long, device=encoded.device
        )

        cum_commitment_loss = 0

        for idx, quantizer in enumerate(self.quantizers):
            output = quantizer(residual)

            quantized = output["quantized"]
            indices = output["indices"]
            commitment_loss = output["commitment_loss"]

            all_indices[:, :, idx] = indices
            cum_commitment_loss = cum_commitment_loss + commitment_loss
            cum_quantized = cum_quantized + quantized
            residual = residual - quantized.detach()

        quantized = encoded + (cum_quantized - encoded).detach()

        return {
            "quantized": quantized,
            "indices": all_indices,
            "commitment_loss": cum_commitment_loss,
        }
