from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    if batch_transforms is None:
        return

    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_loader_config(config, partition):
    if partition == "train":
        return config.dataloader.train
    return config.dataloader.inference


def get_dataloaders(config, device):
    """
    Create dataloaders for all dataset partitions.
    """
    batch_transforms = None

    dataloaders = {}
    for partition in config.datasets.keys():
        dataset = instantiate(config.datasets[partition])
        loader_config = get_loader_config(config, partition)

        dataloaders[partition] = instantiate(
            loader_config,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(partition == "train"),
            shuffle=(partition == "train"),
            worker_init_fn=set_worker_seed,
        )

    return dataloaders, batch_transforms
