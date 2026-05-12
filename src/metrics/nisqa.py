from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

from src.metrics.base_metric import BaseMetric


class NISQAMetric(BaseMetric):
    def __init__(self, sr, *args, **kwargs):
        super().__init__(name="NISQA", *args, **kwargs)

        self.metric = NonIntrusiveSpeechQualityAssessment(fs=sr)

    def __call__(self, generated, **batch):
        generated = generated.detach().cpu().squeeze(1)
        value = self.metric(generated)
        mos = value[0]

        self.metric.reset()
        return mos.item()
