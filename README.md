# SoundStream для речи

Этот репозиторий содержит реализацию аудиокодека **SoundStream**.

## Что реализовано

- generator:
  - encoder;
  - residual vector quantizer;
  - decoder.
- discriminator.
- Обучение в двух режимах:
  - GAN-режим;
  - no_gan-режим.
- Метрики:
  - STOI;
  - NISQA;
  - codebook perplexity.
- Анализ:
  - in-domain (`LibriSpeech`);
  - out-domain English / Russian (`Common Voice`);

## Структура

Основные папки и файлы:

- `train.py` - запуск обучения.
- `src/models/` - модель SoundStream и discriminator.
- `src/loss/` - функции потерь.
- `src/datasets/` - датасеты и collate-функции.
- `src/trainer/` - trainer.
- `src/configs/` - Hydra-конфиги.
- `analysis/` - код для анализа на инференсе.
- `demo.ipynb` - demo для инференса на пользовательском аудио.

## Установка

Рекомендуемая версия Python: `3.12.5` (модель писалась и обучалась на ней).

Сначала можно клонировать репозиторий:

```bash
git clone https://github.com/Zagonov/dlr-soundstream.git
cd dlr-soundstream
```

После этого установить зависимости:

```bash
pip install -r requirements.txt
```

## Данные

В основном эксперименте использовались:
- train: `LibriSpeech/train-clean-100`
- validation / final eval: `LibriSpeech/test-clean`

Если `data_dir` не задан внутри конфига (`configs/datasets/librispeech.yaml`), датасет использует стандартную папку внутри репозитория.

## Обучение

### Основной эксперимент

Основной конфиг

```text
src/configs/baseline.yaml
```

Запуск:

```bash
python train.py --config-name baseline
```

### Эксперимент без GAN

Конфиг:

```text
src/configs/no_gan.yaml
```

Запуск:

```bash
python train.py --config-name no_gan
```

В этом режиме:
- `use_gan = False`;
- discriminator не создаётся и не обучается;
- в generator loss отключены adversarial и feature loss.

## Логи и чекпоинты

По умолчанию логирование идёт в Comet ML через:

```text
src/configs/writer/cometml.yaml
```

Чекпоинты обучения сохраняются в:

```text
saved/<run_name>/
```

Если в конфиге задан `monitor`, то после обучения сохраняется лучший чекпоинт:

```text
saved/soundstream-1/model_best.pth
```

## Как воспроизвести модель

Моя финальная модель соответствует конфигу `baseline.yaml`. Чтобы воспроизвести обучение, необходимо:

1. установить зависимости:

```bash
pip install -r requirements.txt
```

2. запустить обучение:

```bash
python train.py --config-name baseline
```

3. после обучения использовать лучший чекпоинт:

```text
saved/soundstream-1/model_best.pth
```

## Скачивание чекпоинта

Итоговый чекпоинт можно скачать из HugginFace по ссылке (в демо реализовано через библиотеку `wget`):

```bash
mkdir -p saved
wget -O saved/model_best.pth https://huggingface.co/Lunfus/soundstream/resolve/main/model_best.pth
```

После этого файл будет сохранён в:

```text
saved/model_best.pth
```

## Inference

Основная логика инференса вынесена в:

```text
inference.py
```

Модель можно прогнать на локальном аудиофайле так:

```python
from inference import run_codec

result = run_codec(
    input_path="examples/LJ025-0076.wav",
    checkpoint_path="saved/model_best.pth",
    output_path="examples/reconstructed.wav"
)
```

В `result` возвращаются:
- `original_audio`
- `reconstructed_audio`
- `sample_rate`
- `indices`

## Demo

Для демонстрации работы используется:

```text
demo.ipynb
```

В ноутбуке делается следующее:
1. устанавливает зависимости;
2. клонирует репозиторий;
3. скачивает чекпоинт;
4. скачивает внешний аудиофайл по пользовательской ссылке;
5. прогоняет его через codec (в том числе модель выдаёт индексы из кодбука);
6. воспроизводит оригинал и восстановленное аудио.

## Анализ

### Qualitative analysis

Основные функции находятся в:
- `analysis/in_domain.py`
- `analysis/other_domains.py`
- `analysis/common.py`

Пример использования:

```python
from analysis import show_in_domain_audio_pairs
show_in_domain_audio_pairs()
```

### Quantitative analysis

Основные функции находятся в:

- `analysis/quantitative.py`

В нём подсчитываются:
- RMS
- peak_abs
- zero-crossing rate
- spectral centroid
- spectral flatness
- STOI
- spectral convergence
- SI-SDR
- LSD

Пример использования:

```python
from analysis.quantitative import show_quantitative_summary
show_quantitative_summary(num_examples=100)
```
