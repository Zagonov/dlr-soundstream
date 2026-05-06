import torch
from torch.utils.data import DataLoader

from src.datasets import LibrispeechDataset
from src.loss.reconstruction import ReconstructionLoss
from src.models.soundstream import SoundStream


def match_length(audio, target_length):
    current_length = audio.shape[-1]
    if current_length >= target_length:
        return audio

    pad_length = target_length - current_length
    pad = audio[..., -1:].expand(*audio.shape[:-1], pad_length)
    return torch.cat([audio, pad], dim=-1)


def collate_audio(batch, segment_size):
    audios = [match_length(item["audio"], segment_size) for item in batch]
    return {
        "audio": torch.stack(audios),
        "audio_path": [item["audio_path"] for item in batch],
    }


def main():
    device = "cpu"
    dataset = LibrispeechDataset(
        part="train-clean-100",
        target_sr=16000,
        segment_size=8000,
        limit=6,
        return_spectrogram=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=6,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_audio(items, 8000),
    )
    batch = next(iter(loader))
    audio = batch["audio"].to(device)

    model = SoundStream(n_channels=32, latent_channels=128, strides=[2, 4, 5, 5]).to(
        device
    )
    criterion = ReconstructionLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(),
    # lr=1e-4, betas=(0.5, 0.9))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for iteration in range(1000):
        optimizer.zero_grad()

        outputs = model(audio)
        losses = criterion(generated=outputs["generated"], audio=audio)
        loss = losses["loss"]

        loss.backward()
        optimizer.step()

        print(f"iteration: {iteration}")
        print(f"loss: {loss.item(): .6f}")


if __name__ == "__main__":
    main()
