import torch
import argparse
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lstm'))

from tokenizer import DateTokenizer

def load_model():
    from lstm.model import LSTMDateGenerator
    model = LSTMDateGenerator()
    weights_path = os.path.join(os.path.dirname(__file__), 'lstm', 'weights', 'best.pt')
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(input_path: str, output_path: str) -> None:
    tokenizer = DateTokenizer()
    model     = load_model()

    with open(input_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    results = []
    BATCH   = 256

    for i in range(0, len(lines), BATCH):
        batch_lines = lines[i:i+BATCH]
        batch_conds = []
        clean_lines = []

        for line in batch_lines:
            parts      = line.strip().split()
            clean_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}"
            clean_lines.append(clean_line)
            cond = tokenizer.parse_input_line(clean_line)
            batch_conds.append(list(cond))

        cond_tensor = torch.tensor(batch_conds, dtype=torch.long)

        with torch.no_grad():
            preds = model.generate(cond_tensor)

        for clean_line, pred in zip(clean_lines, preds):
            results.append(f"{clean_line} {pred}")

    with open(output_path, 'w') as f:
        for r in results:
            f.write(r + '\n')

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Input file path')
    parser.add_argument('-o', required=True, help='Output file path')
    args = parser.parse_args()
    predict(args.i, args.o)