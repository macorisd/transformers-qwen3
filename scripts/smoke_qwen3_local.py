#!/usr/bin/env python
"""Minimal Qwen3 sanity checks for this local Transformers checkout."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

DEFAULT_HF_HOME = Path(__file__).resolve().parents[1] / ".hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))

import torch

from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3

ROPE_WAVEFORMS = ("sinusoid", "triangular", "square", "sawtooth")


def run_local_forward(rope_waveform: str) -> None:
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_waveform=rope_waveform,
    )
    model = Qwen3ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)

    print("modeling_qwen3:", inspect.getfile(modeling_qwen3))
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("rope_waveform:", rope_waveform)
    print("logits_shape:", tuple(outputs.logits.shape))
    print("loss_finite:", bool(torch.isfinite(outputs.loss)))


def run_hub_check(model_name: str) -> None:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("hub_model:", model_name)
    print("hub_config:", config.__class__.__name__)
    print("hub_model_type:", config.model_type)
    print("hub_layers:", config.num_hidden_layers)
    print("hub_hidden_size:", config.hidden_size)
    print("hub_attention_heads:", config.num_attention_heads)
    print("hub_tokenizer:", tokenizer.__class__.__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-hub", action="store_true", help="Also fetch config/tokenizer metadata from Hugging Face.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--rope-waveform", default="sinusoid", choices=ROPE_WAVEFORMS)
    parser.add_argument("--all-waves", action="store_true", help="Run the local forward check for all RoPE waveforms.")
    args = parser.parse_args()

    waveforms = ROPE_WAVEFORMS if args.all_waves else (args.rope_waveform,)
    for waveform in waveforms:
        run_local_forward(waveform)
    if args.check_hub:
        run_hub_check(args.model_name)


if __name__ == "__main__":
    main()
