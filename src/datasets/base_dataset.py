import logging
import random

import numpy as np
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        index,
        target_sr=16000,
        segment_size=8000,
        limit=None,
        max_audio_length=None,
        shuffle_index=False,
        instance_transforms=None,
        return_spectrogram=True,
    ):
        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(index, max_audio_length)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)

        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.target_sr = target_sr
        self.segment_size = segment_size
        self.instance_transforms = instance_transforms or {}
        self.return_spectrogram = return_spectrogram

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]

        audio = self.load_audio(audio_path)
        audio = self.crop(audio)

        instance_data = {
            "audio": audio,
            "audio_path": audio_path,
        }

        instance_data = self.preprocess_data(instance_data)

        if self.return_spectrogram and "get_spectrogram" in self.instance_transforms:
            instance_data["spectrogram"] = self.get_spectrogram(instance_data["audio"])

        return instance_data

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)

        # Keep only first channel: [channels, time] -> [1, time]
        audio_tensor = audio_tensor[0:1, :]

        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )

        return audio_tensor

    def crop(self, audio):
        audio_len = audio.shape[-1]

        if audio_len > self.segment_size:
            start = random.randint(0, audio_len - self.segment_size)
            return audio[:, start : start + self.segment_size]

        return audio

    def get_spectrogram(self, audio):
        return self.instance_transforms["get_spectrogram"](audio)

    def preprocess_data(self, instance_data):
        for transform_name, transform in self.instance_transforms.items():
            if transform_name == "get_spectrogram":
                continue

            if transform_name in instance_data:
                instance_data[transform_name] = transform(instance_data[transform_name])

        return instance_data

    @staticmethod
    def _filter_records_from_dataset(index, max_audio_length):
        initial_size = len(index)

        if max_audio_length is None:
            return index

        exceeds_audio_length = (
            np.array([el["audio_len"] for el in index]) >= max_audio_length
        )

        total = exceeds_audio_length.sum()

        if total > 0:
            logger.info(
                f"{total} ({total / initial_size: .1%}) records"
                f" are longer than "
                f"{max_audio_length} seconds. Excluding them."
            )

            index = [
                el for el, exclude in zip(index, exceeds_audio_length) if not exclude
            ]

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, "Each dataset item should include "
            "field 'path' - path to audio file."

            assert "audio_len" in entry, "Each dataset item should include "
            "field 'audio_len' - audio length."

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]

        return index
