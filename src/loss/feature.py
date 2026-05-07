from torch import nn


class FeatureLoss(nn.Module):
    """
    Feature matching loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, generated_feature_maps, target_feature_maps):
        loss = 0
        n_maps = 0

        for gen_disc_maps, target_disc_maps in zip(
            generated_feature_maps, target_feature_maps
        ):
            for gen_map, target_map in zip(gen_disc_maps, target_disc_maps):
                loss += self.loss(gen_map, target_map.detach())
                n_maps += 1

        loss = loss / n_maps

        return {"loss": loss}
