import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker


class Inferencer:
    def __init__(
        self,
        generator,
        config,
        device,
        dataloaders,
        save_path=None,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.generator = generator
        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.device = device
        self.dataloaders = dataloaders
        self.save_path = save_path
        self.batch_transforms = batch_transforms or {"inference": None}

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    @torch.no_grad()
    def process_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        batch.update(self.generator(**batch))

        for metric in self.metrics["inference"]:
            metrics.update(metric.name, metric(**batch), n=batch["audio"].shape[0])

        return batch

    def _inference_part(self, part, dataloader):
        self.generator.eval()
        self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(batch, self.evaluation_metrics)
                if self.save_path is not None:
                    self.save_batch_outputs(part, batch_idx, batch)

        return self.evaluation_metrics.result()

    def save_batch_outputs(self, part, batch_idx, batch):
        for item_idx, generated in enumerate(batch["generated"]):
            output_id = batch_idx * batch["generated"].shape[0] + item_idx
            output = {
                "generated": generated.detach().cpu(),
                "audio_path": batch["audio_path"][item_idx],
            }
            torch.save(output, self.save_path / part / f"output_{output_id}.pth")

    def move_batch_to_device(self, batch):
        for tensor_name in self.cfg_trainer.device_tensors:
            batch[tensor_name] = batch[tensor_name].to(self.device)
        return batch

    def transform_batch(self, batch):
        transforms = self.batch_transforms.get("inference")
        if transforms is not None:
            for transform_name, transform in transforms.items():
                batch[transform_name] = transform(batch[transform_name])
        return batch

    def _from_pretrained(self, pretrained_path):
        checkpoint = torch.load(str(pretrained_path), map_location=self.device)
        if "generator" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator"])
        else:
            self.generator.load_state_dict(checkpoint)
