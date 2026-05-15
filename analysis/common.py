import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from IPython.display import Audio, Markdown, display
from omegaconf import OmegaConf

from src.utils.io_utils import ROOT_PATH

CHECKPOINT_PATH = ROOT_PATH / "saved" / "model_best.pth"
sr = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path=CHECKPOINT_PATH):
    config = OmegaConf.load(ROOT_PATH / "src" / "configs" / "model" / "baseline.yaml")
    model = instantiate(config.generator).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model


def stft(audio):
    window = torch.hann_window(512)
    spec = torch.stft(
        audio,
        n_fft=512,
        hop_length=128,
        win_length=512,
        return_complex=True,
        window=window,
    )
    return torch.log(spec.abs() + 1e-6)


def collect_examples(dataset, num_examples=5, checkpoint_path=CHECKPOINT_PATH):
    model = load_model(checkpoint_path)
    dataset_indices = list(range(0, len(dataset), len(dataset) // num_examples))

    examples = []
    for example_id, dataset_idx in enumerate(dataset_indices):
        item = dataset[dataset_idx]
        audio = item["audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(audio=audio)

        examples.append(
            {
                "example_id": example_id + 1,
                "original": item["audio"][0].cpu(),
                "generated": output["generated"][0, 0].cpu(),
            }
        )

    return examples


def show_audio_pairs(examples):
    for example in examples:
        display(Markdown(f"Пример {example['example_id']}\n"))
        display(Markdown("Оригинал"))
        display(Audio(example["original"].numpy(), rate=sr))
        display(Markdown("Восстановленное"))
        display(Audio(example["generated"].numpy(), rate=sr))


def show_waveforms(examples):
    fig, axes = plt.subplots(len(examples), 2, figsize=(14, 3 * len(examples)))

    for row, example in enumerate(examples):
        original_time = torch.arange(example["original"].numel()) / sr
        generated_time = torch.arange(example["generated"].numel()) / sr

        axes[row, 0].plot(
            original_time.numpy(), example["original"].numpy(), linewidth=1
        )
        axes[row, 0].set_title(f"Пример {example['example_id']} (оригинал)")
        axes[row, 0].set_xlabel("Время, с")
        axes[row, 0].set_ylabel("Амплитуда")

        axes[row, 1].plot(
            generated_time.numpy(), example["generated"].numpy(), linewidth=1
        )
        axes[row, 1].set_title(f"Пример {example['example_id']} (восстановленный)")
        axes[row, 1].set_xlabel("Время, с")
        axes[row, 1].set_ylabel("Амплитуда")

    plt.tight_layout()


def show_stfts(examples):
    fig, axes = plt.subplots(len(examples), 2, figsize=(14, 4 * len(examples)))

    for row, example in enumerate(examples):
        original_spec = stft(example["original"]).numpy()
        generated_spec = stft(example["generated"]).numpy()

        axes[row, 0].imshow(original_spec, aspect="auto", origin="lower", cmap="magma")
        axes[row, 0].set_title(f"Пример {example['example_id']} (оригинал)")
        axes[row, 0].set_xlabel("Фреймы")
        axes[row, 0].set_ylabel("Частоты")

        axes[row, 1].imshow(generated_spec, aspect="auto", origin="lower", cmap="magma")
        axes[row, 1].set_title(f"Пример {example['example_id']} (восстановленный)")
        axes[row, 1].set_xlabel("Фреймы")
        axes[row, 1].set_ylabel("Частоты")

    plt.tight_layout()
