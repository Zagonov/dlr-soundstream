from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    SoundStream trainer
    """

    def process_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        batch.update(self.generator(**batch))

        batch_size = batch["audio"].shape[0]

        if self.is_train:
            self.discriminator_optimizer.zero_grad()
            batch.update(
                self.discriminator(
                    audio=batch["audio"],
                    generated=batch["generated"].detach(),
                )
            )
            batch.update(self.discriminator_criterion(**batch))
            batch["discriminator_loss"].backward()
            self.discriminator_optimizer.step()

            self.generator_optimizer.zero_grad()
            batch.update(self.discriminator(**batch))
            batch.update(self.generator_criterion(**batch))
            batch["generator_loss"].backward()
            self.generator_optimizer.step()

            if self.generator_lr_scheduler is not None:
                self.generator_lr_scheduler.step()
            if self.discriminator_lr_scheduler is not None:
                self.discriminator_lr_scheduler.step()

            metric_funcs = self.metrics["train"]
        else:
            batch.update(self.discriminator(**batch))
            batch.update(self.discriminator_criterion(**batch))
            batch.update(self.generator_criterion(**batch))
            metric_funcs = self.metrics["inference"]

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=batch_size)

        for metric in metric_funcs:
            metrics.update(metric.name, metric(**batch), n=batch_size)

        return batch

    def _log_batch(self, batch, mode="train"):
        sample_rate = self.config.datasets.train.get("target_sr", 16000)
        num_examples = self.config.trainer.get("num_audio_log_examples", 5)
        num_examples = min(num_examples, batch["audio"].shape[0])

        for index in range(num_examples):
            audio = batch["audio"][index].detach().cpu()
            generated = batch["generated"][index].detach().cpu()

            self.writer.add_audio(
                f"target_audio_{index}",
                audio,
                sample_rate=sample_rate,
            )
            self.writer.add_audio(
                f"generated_audio_{index}",
                generated,
                sample_rate=sample_rate,
            )
