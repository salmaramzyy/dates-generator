# Dates Generator — Assignment 2

A conditional date generation project using four deep learning models. Given a set of conditions (day of week, month, leap year, decade), each model generates a valid date that satisfies all conditions.

## Models

| Model | Type | Category |
|-------|------|----------|
| LSTM | Autoregressive sequence generator | From course |
| Conditional GAN | Generator + Discriminator | From course (required) |
| VAE | Variational Autoencoder | Outside course |
| Transformer | Decoder-only seq2seq | Outside course |

## Results

| Model | Valid Date | Month | Leap | Decade | Day | All Pass |
|-------|-----------|-------|------|--------|-----|----------|
| LSTM | 100% | 100% | 100% | 100% | 14.42% | 14.42% |
| GAN | 99.80% | 98.81% | 95.28% | 83.38% | 14.68% | 11.80% |
| VAE | 98.17% | 98.17% | 96.94% | 98.17% | 14.42% | 14.25% |
| Transformer | 100% | 100% | 100% | 100% | 13.75% | 13.75% |

## Project Structure
dates-generator/
├── data/
│   └── data.txt
├── model/
│   ├── shared/
│   │   ├── tokenizer.py
│   │   ├── dataset.py
│   │   └── utils.py
│   ├── lstm/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── weights/
│   │   └── logs/
│   ├── gan/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── weights/
│   │   └── logs/
│   ├── vae/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── weights/
│   │   └── logs/
│   ├── transformer/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── weights/
│   │   └── logs/
│   └── predict.py
└── environment.yml

## Setup

```bash
conda env create -f environment.yml
conda activate dates_gen
```

## Training

```bash
python model/lstm/train.py
python model/gan/train.py
python model/vae/train.py
python model/transformer/train.py
```

## Inference

```bash
python model/predict.py -i data/example_input.txt -o data/predictions.txt
```

## Dataset

146,462 samples covering dates from 1-1-1800 to 31-12-2200. Each sample contains 4 conditions and a valid output date.
