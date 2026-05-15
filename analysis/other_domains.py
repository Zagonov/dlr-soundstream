import io
from itertools import islice

import torchaudio
from torch.utils.data import Dataset

from analysis.common import (
    CHECKPOINT_PATH,
    collect_examples,
    show_audio_pairs,
    show_stfts,
    show_waveforms,
)
from datasets import Audio, load_dataset


class HFDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        audio_bytes = row["audio"]["bytes"]
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        waveform = waveform[:1]

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=16000,
            )
            sample_rate = 16000

        return {"audio": waveform}


def load_ru_examples(num_examples=5):
    dataset = load_dataset(
        "fixie-ai/common_voice_17_0",
        "ru",
        split="test",
        streaming=True,
    )

    dataset = dataset.cast_column("audio", Audio(decode=False))
    rows = []
    for i, row in enumerate(dataset):
        if i >= num_examples:
            break
        rows.append(row)
    dataset = HFDataset(rows)

    return collect_examples(dataset, num_examples=num_examples)


def load_en_examples(num_examples=5):
    dataset = load_dataset(
        "fixie-ai/common_voice_17_0",
        "en",
        split="test",
        streaming=True,
    )

    dataset = dataset.cast_column("audio", Audio(decode=False))
    rows = []
    for i, row in enumerate(dataset):
        if i >= num_examples:
            break
        rows.append(row)
    dataset = HFDataset(rows)

    return collect_examples(dataset, num_examples=num_examples)


def show_ru_domain_audio_pairs(num_examples=5):
    examples = load_ru_examples(num_examples=num_examples)
    show_audio_pairs(examples)


def show_ru_domain_waveforms(num_examples=5):
    examples = load_ru_examples(num_examples=num_examples)
    show_waveforms(examples)


def show_ru_domain_stfts(num_examples=5):
    examples = load_ru_examples(num_examples=num_examples)
    show_stfts(examples)


def show_en_domain_audio_pairs(num_examples=5):
    examples = load_en_examples(num_examples=num_examples)
    show_audio_pairs(examples)


def show_en_domain_waveforms(num_examples=5):
    examples = load_en_examples(num_examples=num_examples)
    show_waveforms(examples)


def show_en_domain_stfts(num_examples=5):
    examples = load_en_examples(num_examples=num_examples)
    show_stfts(examples)
