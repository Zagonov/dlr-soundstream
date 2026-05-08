from torch import nn


class GeneratorAdvLoss(nn.Module):
    """
    Adversarial loss for the generator

    Принимает списки из K тензоров, каждый тензор B x T_k
    """

    def __init__(self):
        super().__init__()

    def forward(self, generated_logits):
        """ """
        loss = 0
        for gen_logit in generated_logits:
            gen_loss = (1 - gen_logit).clip(0).mean()
            loss += gen_loss

        loss = loss / len(generated_logits)

        return {"loss": loss}
