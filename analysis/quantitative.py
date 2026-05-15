import pandas as pd
import torch
from IPython.display import Markdown, display
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm.auto import tqdm

from analysis.common import sr, stft
from analysis.in_domain import load_examples as load_in_examples
from analysis.other_domains import load_en_examples, load_ru_examples


def rms(audio):
    return torch.sqrt(torch.mean(audio**2)).item()


def peak_abs(audio):
    return audio.abs().max().item()


def zero_crossing_rate(audio):
    if audio.numel() < 2:
        return 0.0
    return ((audio[:-1] * audio[1:]) < 0).float().mean().item()


def spectral_stats(audio):
    log_spec = stft(audio)
    spec = torch.exp(log_spec)
    freqs = torch.linspace(0, sr / 2, spec.shape[0], dtype=spec.dtype)

    centroid = ((spec * freqs[:, None]).sum() / (spec.sum() + 1e-6)).item()
    geometric_mean = torch.exp(log_spec.mean())
    arithmetic_mean = torch.mean(spec)
    flatness = (geometric_mean / (arithmetic_mean + 1e-6)).item()

    return {
        "log_spec": log_spec,
        "spec": spec,
        "spectral_centroid": centroid,
        "spectral_flatness": flatness,
    }


def spectral_corvegence(original_stats, generated_stats):
    numerator = torch.linalg.vector_norm(
        original_stats["spec"] - generated_stats["spec"]
    )
    denominator = torch.linalg.vector_norm(original_stats["spec"]) + 1e-6

    return {"spectral_convergence": (numerator / denominator).item()}


def si_sdr(original, generated):
    reference = original - original.mean()
    est = generated - generated.mean()

    target = torch.dot(est, reference) / (torch.dot(reference, reference) + 1e-6)
    target = target * reference
    noise = est - target

    div = torch.sum(target**2) / (torch.sum(noise**2) + 1e-6)
    return 10 * torch.log10(div + 1e-6).item()


def lsd(original_stats, generated_stats):
    sq_error = (original_stats["log_spec"] - generated_stats["log_spec"]) ** 2
    return torch.sqrt(torch.mean(sq_error, dim=0)).mean().item()


def stoi_score(original, generated, metric):
    value = metric(generated.unsqueeze(0), original.unsqueeze(0))
    metric.reset()
    return value.item()


def stats(audio, prefix, spectral):
    return {
        f"{prefix}_rms": rms(audio),
        f"{prefix}_peak_abs": peak_abs(audio),
        f"{prefix}_zcr": zero_crossing_rate(audio),
        f"{prefix}_spectral_centroid": spectral["spectral_centroid"],
        f"{prefix}_spectral_flatness": spectral["spectral_flatness"],
    }


def compute_stats(examples):
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=sr)

    rows = []
    for example in tqdm(examples, total=len(examples), leave=False):
        original = example["original"]
        generated = example["generated"]
        original_spectral = spectral_stats(original)
        generated_spectral = spectral_stats(generated)

        row = {"example_id": example["example_id"]}
        row.update(stats(original, "original", original_spectral))
        row.update(stats(generated, "generated", generated_spectral))
        row["stoi"] = stoi_score(original, generated, stoi_metric)
        row.update(spectral_corvegence(original_spectral, generated_spectral))
        row["si_sdr"] = si_sdr(original, generated)
        row["lsd"] = lsd(original_spectral, generated_spectral)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_stats(rows):
    distribution_metrics = [
        "rms",
        "peak_abs",
        "zcr",
        "spectral_centroid",
        "spectral_flatness",
    ]

    summary_rows = []
    for metric in distribution_metrics:
        original_col = f"original_{metric}"
        generated_col = f"generated_{metric}"
        delta = rows[generated_col] - rows[original_col]

        summary_rows.append(
            {
                "metric": metric,
                "original_mean": rows[original_col].mean(),
                "generated_mean": rows[generated_col].mean(),
                "delta_mean": delta.mean(),
            }
        )

    pair_summary = pd.DataFrame(
        [
            {"metric": "stoi", "mean": rows["stoi"].mean()},
            {
                "metric": "spectral_convergence",
                "mean": rows["spectral_convergence"].mean(),
            },
            {"metric": "si_sdr", "mean": rows["si_sdr"].mean()},
            {"metric": "lsd", "mean": rows["lsd"].mean()},
        ]
    )

    return pd.DataFrame(summary_rows), pair_summary


def show_quantitative_summary(num_examples=5):
    examples = load_in_examples(num_examples)
    rows = compute_stats(examples)
    summary, pair_summary = summarize_stats(rows)

    display(Markdown("Одиночные метрики"))
    display(summary)
    display(Markdown("Сравнительные метрики"))
    display(pair_summary)


def show_quantitative_en_summary(num_examples=5):
    examples = load_en_examples(num_examples)
    rows = compute_stats(examples)
    summary, pair_summary = summarize_stats(rows)

    display(Markdown("Одиночные метрики"))
    display(summary)
    display(Markdown("Сравнительные метрики"))
    display(pair_summary)


def show_quantitative_ru_summary(num_examples=5):
    examples = load_ru_examples(num_examples)
    rows = compute_stats(examples)
    summary, pair_summary = summarize_stats(rows)

    display(Markdown("Одиночные метрики"))
    display(summary)
    display(Markdown("Сравнительные метрики"))
    display(pair_summary)
