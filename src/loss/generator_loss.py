from torch import nn

from src.loss.feature_loss import FeatureLoss
from src.loss.generator_adv_loss import GeneratorAdvLoss
from src.loss.reconstruction_loss import ReconstructionLoss


class GeneratorLoss(nn.Module):
    def __init__(
        self, lambda_rec=1, lambda_adv=1, lambda_feat=100, lambda_commitment=1
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_commitment = lambda_commitment
        self.reconstruction_loss = ReconstructionLoss()
        self.adversarial_loss = GeneratorAdvLoss()
        self.feature_loss = FeatureLoss()

    def forward(
        self,
        generated,
        audio,
        commitment_loss,
        generated_logits=None,
        generated_feature_maps=None,
        target_feature_maps=None,
        **batch
    ):
        rec = self.reconstruction_loss(generated, audio)["loss"]
        zero = rec.new_zeros(())
        adv = zero
        feat = zero

        if self.lambda_adv != 0:
            adv = self.adversarial_loss(generated_logits)["loss"]
        if self.lambda_feat != 0:
            feat = self.feature_loss(generated_feature_maps, target_feature_maps)[
                "loss"
            ]

        commitment = commitment_loss

        loss = (
            self.lambda_rec * rec
            + self.lambda_adv * adv
            + self.lambda_feat * feat
            + self.lambda_commitment * commitment
        )

        return {
            "generator_loss": loss,
            "reconstruction_loss": rec,
            "generator_adv_loss": adv,
            "feature_loss": feat,
        }
