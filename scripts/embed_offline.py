#!/usr/bin/env python3
"""Precompute T5 token ids and embeddings for the closed instruction set.

Example:
  python scripts/embed_offline.py \
    --infile multi_object_fetch/language/sentences.txt \
    --outfile multi_object_fetch/language/embeds.pkl \
    --model t5
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def embed_t5(sentences, outf: Path) -> None:
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-small", use_legacy=False)
    model = T5EncoderModel.from_pretrained("t5-small")
    model.eval()

    token_cache = {}
    embed_cache = {}
    token_str_cache = {}
    lengths = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        ids = tokens["input_ids"].cpu().numpy()[0]
        with torch.no_grad():
            embed = model(**tokens).last_hidden_state.squeeze(0).cpu().numpy()
        token_cache[sentence] = ids
        embed_cache[sentence] = embed
        # Per-token surface forms with leading spaces preserved via prefix diffs.
        prefixes = [tokenizer.decode(ids[: i + 1]) for i in range(len(ids))]
        pieces = []
        prev = ""
        for full in prefixes:
            if full.startswith(prev):
                pieces.append(full[len(prev) :])
            else:
                pieces.append(full)
            prev = full
        token_str_cache[sentence] = pieces
        lengths.append(len(ids))

    token_cache["<pad>"] = tokenizer.pad_token_id
    token_str_cache["<pad>"] = ["<pad>"]
    with torch.no_grad():
        embed_cache["<pad>"] = (
            model(**tokenizer("<pad>", add_special_tokens=False, return_tensors="pt"))
            .last_hidden_state[0][0]
            .cpu()
            .numpy()
        )
    assert embed_cache["<pad>"].shape == (model.config.d_model,), embed_cache["<pad>"].shape

    outf.parent.mkdir(parents=True, exist_ok=True)
    with open(outf, "wb") as f:
        pickle.dump((token_cache, embed_cache, token_str_cache), f)
    print(f"Wrote {len(token_cache)} entries to {outf}")
    print("token lengths:", lengths)
    print("max len:", max(lengths) if lengths else 0)
    for sentence, pieces in token_str_cache.items():
        if sentence != "<pad>":
            print(f"  {sentence!r} -> {pieces}")



def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        type=Path,
        default=root / "multi_object_fetch" / "language" / "sentences.txt",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=root / "multi_object_fetch" / "language" / "embeds.pkl",
    )
    parser.add_argument("--model", choices=["t5"], default="t5")
    args = parser.parse_args()

    with open(args.infile) as f:
        sentences = [line.strip() for line in f if line.strip()]

    if args.model == "t5":
        embed_t5(sentences, args.outfile)
    else:
        raise ValueError(args.model)


if __name__ == "__main__":
    main()
