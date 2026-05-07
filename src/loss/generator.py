from torch import nn

from src.loss.feature import FeatureLoss
from src.loss.gan import GeneratorAdvLoss
from src.loss.reconstruction import ReconstructionLoss


class GeneratorLoss(nn.Module):
    def __init__(self, lambda_rec=1, lambda_adv=1, lambda_feat=100):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.reconstruction_loss = ReconstructionLoss()
        self.adversarial_loss = GeneratorAdvLoss()
        self.feature_loss = FeatureLoss()

    def forward(
        self,
        generated,
        target,
        generated_logits,
        generated_feature_maps,
        target_feature_maps,
    ):
        rec = self.reconstruction_loss(generated, target)["loss"]
        adv = self.adversarial_loss(generated_logits)["loss"]
        feat = self.feature_loss(generated_feature_maps, target_feature_maps)["loss"]

        loss = self.lambda_rec * rec + self.lambda_adv * adv + self.lambda_feat * feat

        return {
            "loss": loss,
            "reconstruction_loss": rec,
            "generator_adv_loss": adv,
            "feature_loss": feat,
        }
