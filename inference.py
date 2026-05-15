from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf

target_sr = 16000


def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = OmegaConf.load("src/configs/model/baseline.yaml")
    model = instantiate(model_cfg.generator).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model, device


def run_codec(input_path, checkpoint_path, output_path):
    model, device = load_model(checkpoint_path)

    audio, sr = torchaudio.load(input_path)
    audio = audio[:1, :]
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr

    audio = audio.unsqueeze(0).to(device)

    with torch.no_grad():
        indices = model.get_indices(audio)
        reconstructed = model.decode_from_indices(indices)

    original_audio = audio.squeeze(0, 1).detach().cpu()
    reconstructed_audio = reconstructed.squeeze(0, 1).detach().cpu()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, reconstructed_audio.numpy(), sr)

    return {
        "original_audio": original_audio,
        "reconstructed_audio": reconstructed_audio,
        "sample_rate": sr,
        "indices": indices.detach().cpu(),
    }
