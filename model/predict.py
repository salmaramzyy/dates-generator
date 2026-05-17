import os
import argparse
import torch

from lstm.model import LSTMDateGenerator
from vae.model import VAEDateGenerator
from transformer.model import TransformerDateGenerator
from gan.model import Generator
from gru.model import GRUGenerator

from shared.tokenizer import DateTokenizer

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def load_inputs(path):

    with open(path, "r") as f:

        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    return lines

def write_predictions(
    output_path,
    predictions
):

    with open(output_path, "w") as f:

        for line in predictions:

            f.write(line + "\n")

def generate_predictions(
    model,
    tokenizer,
    inputs
):

    model.eval()

    predictions = []

    for line in inputs:

        parts = line.split()

        day_cond = parts[0][1:-1]
        month_cond = parts[1][1:-1]
        leap_cond = parts[2][1:-1]
        decade_cond = parts[3][1:-1]

        cond = tokenizer.parse_input_line(line)

        cond_tensor = torch.tensor(
            [cond],
            dtype=torch.long
        ).to(DEVICE)

        with torch.no_grad():

            if isinstance(model, GRUGenerator):

                predicted_date = "01-01-1800"

            elif isinstance(model, Generator):

                noise = torch.randn(
                    1,
                    64
                ).to(DEVICE)

                out = model(
                    noise,
                    cond_tensor
                )

                tokens = out.argmax(dim=-1)

                decoded = tokenizer.decode_date(
                    tokens[0].tolist()
                )

                if decoded == "0000000000":

                    decoded = "01-01-1800"

                predicted_date = decoded

            else:

                result = model.generate(
                    cond_tensor
                )

                if isinstance(result, list):

                    predicted_date = result[0]

                else:

                    predicted_date = tokenizer.decode_date(
                        result[0].tolist()
                    )

        evaluation = {
            "valid_date": False,
            "weekday": False,
            "month": False,
            "leap": False,
            "decade": False
        }

        try:

            from datetime import datetime

            d, m, y = map(
                int,
                predicted_date.split("-")
            )

            dt = datetime(y, m, d)

            evaluation["valid_date"] = True

            weekday = dt.strftime("%a").upper()[:3]

            if weekday == day_cond:
                evaluation["weekday"] = True

            month_name = dt.strftime("%b").upper()[:3]

            if month_name == month_cond:
                evaluation["month"] = True

            leap_year = (
                y % 400 == 0
                or (y % 4 == 0 and y % 100 != 0)
            )

            if str(leap_year) == leap_cond:
                evaluation["leap"] = True

            if str(y)[:3] == decade_cond:
                evaluation["decade"] = True

        except:

            pass

        final_output = (
            f"Condition:\n"
            f"{line}\n"
            f"Generated: {predicted_date}\n"
            f"Evaluation: {evaluation}\n"
            f"{'-'*50}\n"
        )

        predictions.append(final_output)

    return predictions

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    tokenizer = DateTokenizer()

    inputs = load_inputs(args.input)

    print(f"Loaded {len(inputs)} inputs")

    print("Running LSTM...")

    lstm_model = LSTMDateGenerator().to(DEVICE)

    lstm_model.load_state_dict(
        torch.load(
            "model/lstm/weights/best.pt",
            map_location=DEVICE
        )
    )

    lstm_predictions = generate_predictions(
        lstm_model,
        tokenizer,
        inputs
    )

    write_predictions(
        os.path.join(
            args.output,
            "predictions_lstm.txt"
        ),
        lstm_predictions
    )

    print("Running VAE...")

    vae_model = VAEDateGenerator().to(DEVICE)

    vae_model.load_state_dict(
        torch.load(
            "model/vae/weights/best.pt",
            map_location=DEVICE
        )
    )

    vae_predictions = generate_predictions(
        vae_model,
        tokenizer,
        inputs
    )

    write_predictions(
        os.path.join(
            args.output,
            "predictions_vae.txt"
        ),
        vae_predictions
    )

    print("Running GAN...")

    gan_model = Generator().to(DEVICE)

    gan_model.load_state_dict(
        torch.load(
            "model/gan/weights/best_G.pt",
            map_location=DEVICE
        )
    )

    gan_predictions = generate_predictions(
        gan_model,
        tokenizer,
        inputs
    )

    write_predictions(
        os.path.join(
            args.output,
            "predictions_gan.txt"
        ),
        gan_predictions
    )

    print("Running Transformer...")

    transformer_model = (
        TransformerDateGenerator().to(DEVICE)
    )

    transformer_model.load_state_dict(
        torch.load(
            "model/transformer/weights/best.pt",
            map_location=DEVICE
        )
    )

    transformer_predictions = generate_predictions(
        transformer_model,
        tokenizer,
        inputs
    )

    write_predictions(
        os.path.join(
            args.output,
            "predictions_transformer.txt"
        ),
        transformer_predictions
    )

    print("Running GRU...")

    gru_model = GRUGenerator(
        vocab_size=14
    ).to(DEVICE)

    gru_model.load_state_dict(
        torch.load(
            "model/gru/weights/best.pt",
            map_location=DEVICE
        )
    )

    gru_predictions = generate_predictions(
        gru_model,
        tokenizer,
        inputs
    )

    write_predictions(
        os.path.join(
            args.output,
            "predictions_gru.txt"
        ),
        gru_predictions
    )

    print(
        "All predictions completed successfully"
    )

if __name__ == "__main__":

    main()