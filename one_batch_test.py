import torch
from torch.utils.data import DataLoader

from src.datasets.libra_dataset import LibrispeechDataset
from src.loss import DiscriminatorLoss, GeneratorLoss
from src.metrics import CodebookPerplexityMetric, NISQAMetric, STOIMetric
from src.models import Discriminator, SoundStream


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
    batch["audio"] = batch["audio"].to(device)

    generator = SoundStream(
        n_channels=32,
        latent_channels=128,
        strides=[2, 4, 5, 5],
        num_quantizers=8,
        codebook_size=128,
        gamma=0.99,
        kmeans_n_iters=20,
        code_threshold=2,
    ).to(device)

    discriminator = Discriminator().to(device)

    g_criterion = GeneratorLoss().to(device)
    d_criterion = DiscriminatorLoss().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9)
    )

    stoi = STOIMetric(sr=16000)
    nisqa = NISQAMetric(sr=16000)
    perplexity = CodebookPerplexityMetric(codebook_size=128)

    generator.train()
    discriminator.train()

    for iteration in range(1000):
        g_output = generator(**batch)
        batch.update(g_output)

        metrics = {
            "STOI": stoi(**batch),
            "NISQA": nisqa(**batch),
            "Perplexity": perplexity(**batch),
        }

        batch.update(metrics)

        # обучение дискриминатора
        d_optimizer.zero_grad()
        d_output = discriminator(
            audio=batch["audio"], generated=batch["generated"].detach()
        )
        batch.update(d_output)
        d_loss = d_criterion(**batch)
        batch.update(d_loss)
        d_loss["discriminator_loss"].backward()
        d_optimizer.step()

        # обучение генератора
        g_optimizer.zero_grad()
        d_output = discriminator(**batch)
        batch.update(d_output)
        g_loss = g_criterion(**batch)
        batch.update(g_loss)
        g_loss["generator_loss"].backward()
        g_optimizer.step()

        print(f"iteration: {iteration}")
        print(f"g_loss: {batch['generator_loss']}")
        print(f"d_loss: {batch['discriminator_loss']}")
        print(f"gen_adv_loss: {batch['generator_adv_loss']}")
        print(f"feature_loss: {batch['feature_loss']}")
        print(f"reconstruction_loss: {batch['reconstruction_loss']}")
        print(f"commitment_loss: {batch['commitment_loss']}")
        print(f"STOI: {batch['STOI']}")
        print(f"NISQA: {batch['NISQA']}")
        print(f"Perplexity: {batch['Perplexity']}")


if __name__ == "__main__":
    main()
