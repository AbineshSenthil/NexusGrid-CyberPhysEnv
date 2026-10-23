#!/usr/bin/env python3
"""
Create a tiny random-weight causal LM with a NexusGrid-trained BPE tokenizer.

Used for CPU smoke tests of the GRPO pipeline without downloading a model:

    python training/make_tiny_model.py --out runs/tiny-model
    python training/train_grpo.py --model runs/tiny-model --cpu --max-steps 2 --dataset-episodes 4 \\
        --num-generations 2 --max-completion-length 16 --out runs/grpo-smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/tiny-model")
    args = p.parse_args(argv)

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from inference import SYSTEM_PROMPT, render_observation
    from server.nexusgrid_environment import NexusgridEnvironment

    env = NexusgridEnvironment()
    corpus = [SYSTEM_PROMPT]
    for task in range(6):
        corpus.append(render_observation(env.reset(seed=task, task_id=task)))
    corpus.append('{"action_type":"dispatch_generation","node_id":"NODE_01","mw":120}')

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=["<unk>", "<pad>", "<|end|>", "<|system|>",
                                                                   "<|user|>", "<|assistant|>"])
    tok.train_from_iterator(corpus, trainer)
    from tokenizers import decoders

    tok.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
                                        eos_token="<|end|>")
    tokenizer.chat_template = CHAT_TEMPLATE
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                         bos_token_id=tokenizer.eos_token_id)
    model = LlamaForCausalLM(config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"tiny model ({sum(p.numel() for p in model.parameters()) / 1e3:.0f}k params) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
