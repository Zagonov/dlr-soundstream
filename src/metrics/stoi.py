from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, sr, *args, **kwargs):
        super().__init__(name="STOI", *args, **kwargs)

        self.metric = ShortTimeObjectiveIntelligibility(fs=sr)

    def __call__(self, generated, audio, **batch):
        generated = generated.squeeze(1)
        audio = audio.squeeze(1)
        value = self.metric(generated, audio)

        return value.item()
