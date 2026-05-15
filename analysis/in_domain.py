from analysis.common import (
    CHECKPOINT_PATH,
    collect_examples,
    show_audio_pairs,
    show_stfts,
    show_waveforms,
    sr,
)
from src.datasets import LibrispeechDataset
from src.utils.io_utils import ROOT_PATH

INDEX_DIR = ROOT_PATH / "saved" / "indices" / "librispeech"


def load_examples(num_examples=5):
    dataset = LibrispeechDataset(
        part="test-clean",
        index_dir=INDEX_DIR,
        target_sr=sr,
        segment_size=None,
        shuffle_index=False,
    )
    return collect_examples(dataset, num_examples=num_examples)


def show_in_domain_audio_pairs(num_examples=5):
    examples = load_examples(num_examples=num_examples)
    show_audio_pairs(examples)


def show_in_domain_waveforms(num_examples=5):
    examples = load_examples(num_examples=num_examples)
    show_waveforms(examples)


def show_in_domain_stfts(num_examples=5):
    examples = load_examples(num_examples=num_examples)
    show_stfts(examples)
