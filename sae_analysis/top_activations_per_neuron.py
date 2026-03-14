"""
Load all sae_distribution_*.json files, then for each SAE neuron (concept in 16K space)
retrieve the indices of the 10 sample records that activated it the most.
Writes results to a single JSON file for inspection.
"""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Top-10 sample indices per SAE neuron")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="controls",
        help="Directory containing sae_distribution_*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sae_analysis/neuron_top10_samples.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top sample indices per neuron (default: 10)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/home/z5517269/Semantic-SAE/data/nmt/test.txt",
        help="Path to test file (one sample per line); line index = sample index",
    )
    args = parser.parse_args()

    # Load test samples: line i (0-based) = sample index i
    test_lines = []
    if os.path.isfile(args.test_file):
        with open(args.test_file, "r", encoding="utf-8") as f:
            test_lines = [line.split('\t')[0].rstrip("\n") for line in f]
    else:
        print(f"Warning: test_file not found {args.test_file}, skipping sample text.")

    pattern = os.path.join(args.input_dir, "sae_distribution_*.json")
    paths = sorted(glob.glob(pattern), key=lambda p: int(p.split("_")[-1].replace(".json", "")))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern}")

    # Load all: list of (index, latent_vec) with latent_vec as list of floats
    samples = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        idx = data["index"]
        latent = [float(x) for x in data["latent"]]
        samples.append((idx, latent))

    # d_sae from first sample
    d_sae = len(samples[0][1])
    num_samples = len(samples)

    # For each neuron j: top-k sample indices, average activation, and test sample text
    # samples[i] = (original_index, list of d_sae values)
    result = {}
    avg_activation = {}
    top_samples_text = {}
    for j in range(d_sae):
        values = [s[1][j] for s in samples]
        avg_activation[f"neuron_{j}"] = round(sum(values) / len(values), 10)
        # (value, original_sample_index) for each sample
        values_and_indices = [(s[1][j], s[0]) for s in samples]
        values_and_indices.sort(reverse=True, key=lambda x: x[0])
        top_indices = [values_and_indices[i][1] for i in range(min(args.top_k, len(values_and_indices)))]
        result[f"neuron_{j}"] = top_indices
        # Look up test file line for each index (line index = sample index)
        if test_lines:
            top_samples_text[f"neuron_{j}"] = [
                test_lines[idx] if 0 <= idx < len(test_lines) else ""
                for idx in top_indices
            ]
        else:
            top_samples_text[f"neuron_{j}"] = []

    out_data = {
        "d_sae": d_sae,
        "num_samples": num_samples,
        "top_k": args.top_k,
        "input_dir": args.input_dir,
        "test_file": args.test_file,
        "neurons": result,
        "neuron_avg_activation": avg_activation,
        "neuron_top_samples_text": top_samples_text,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}: {d_sae} neurons, top-{args.top_k} sample indices each (from {num_samples} samples).")


if __name__ == "__main__":
    main()
