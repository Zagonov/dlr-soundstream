import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data loader instances
    dataloaders, batch_transforms = get_dataloaders(config, device)
    use_gan = config.trainer.get("use_gan", True)

    # build model architectures, then print to console
    generator = instantiate(config.model.generator).to(device)
    discriminator = None
    if use_gan:
        discriminator = instantiate(config.model.discriminator).to(device)
    logger.info(generator)
    if discriminator is not None:
        logger.info(discriminator)

    # get function handles of loss and metrics
    generator_loss = instantiate(config.loss_function.generator).to(device)
    discriminator_loss = None
    if use_gan:
        discriminator_loss = instantiate(config.loss_function.discriminator).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    # build optimizer, learning rate scheduler

    generator_optimizer = instantiate(
        config.optimizer.generator,
        params=generator.parameters(),
    )
    discriminator_optimizer = None
    if use_gan:
        discriminator_optimizer = instantiate(
            config.optimizer.discriminator,
            params=discriminator.parameters(),
        )

    generator_lr_scheduler = None
    discriminator_lr_scheduler = None
    if config.get("lr_scheduler") is not None:
        if config.lr_scheduler.get("generator") is not None:
            generator_lr_scheduler = instantiate(
                config.lr_scheduler.generator,
                optimizer=generator_optimizer,
            )
        if use_gan and config.lr_scheduler.get("discriminator") is not None:
            discriminator_lr_scheduler = instantiate(
                config.lr_scheduler.discriminator,
                optimizer=discriminator_optimizer,
            )

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        generator_criterion=generator_loss,
        discriminator_criterion=discriminator_loss,
        metrics=metrics,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_lr_scheduler=generator_lr_scheduler,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
    )

    trainer.train()


if __name__ == "__main__":
    main()
