import torch


def match_length(audio, target_length):
    current_length = audio.shape[-1]
    if current_length == target_length:
        return audio

    if current_length > target_length:
        return audio[..., :target_length]

    pad_length = target_length - current_length
    pad = audio[..., -1:].expand(*audio.shape[:-1], pad_length)
    return torch.cat([audio, pad], dim=-1)


def collate_fn(dataset):
    target_length = max(item["audio"].shape[-1] for item in dataset)
    audios = [match_length(item["audio"], target_length) for item in dataset]

    return {
        "audio": torch.stack(audios),
        "audio_path": [item["audio_path"] for item in dataset],
    }
