#!/usr/bin/env python3
"""
Evaluate a local Hugging Face model (optionally with a LoRA adapter) as a pure
NexusGrid operator on held-out seeds. No fallback: invalid outputs become
``advance_tick`` and are counted.

    python training/eval_llm_local.py --model Qwen/Qwen3-1.7B --seeds 10 --out results/llm/qwen3-1.7b-base
    python training/eval_llm_local.py --model Qwen/Qwen3-1.7B --adapter runs/grpo/qwen3-1.7b/adapter \\
        --seeds 10 --out results/llm/qwen3-1.7b-grpo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import SYSTEM_PROMPT, LLMPolicy  # noqa: E402
from scripts.evaluate import markdown_table, summarise  # noqa: E402
from server.nexusgrid_environment import NexusgridEnvironment  # noqa: E402
from server.scenarios import EVAL_SEED_START, TASK_NAMES  # noqa: E402


class HFLocalPolicy(LLMPolicy):
    """LLMPolicy that generates with a local transformers (or Unsloth) model instead of an HTTP endpoint."""

    def __init__(self, model_name: str, adapter: str | None = None, load_in_4bit: bool = False,
                 max_new_tokens: int = 64, backend: str = "hf"):
        super().__init__(model=model_name if not adapter else f"{model_name}+{Path(adapter).name}",
                         api_base="local", client=object())
        import torch

        self.torch = torch
        if backend == "unsloth":
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter or model_name, max_seq_length=2048, load_in_4bit=load_in_4bit or "bnb" in model_name)
            FastLanguageModel.for_inference(model)
            self.tokenizer, self.hf_model = tokenizer, model
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from training.train_grpo import dtype_kwargs

            self.tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
            kwargs = dtype_kwargs(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            if torch.cuda.is_available():
                kwargs["device_map"] = {"": 0}
                if load_in_4bit:
                    from transformers import BitsAndBytesConfig

                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            if adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter)
            self.hf_model = model.eval()
        self.max_new_tokens = max_new_tokens
        from training.logutil import quiet_transformers

        quiet_transformers()

    def complete(self, prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.hf_model.device)
        with self.torch.no_grad():
            output = self.hf_model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--tasks", default="0,1,2,3,4,5")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--backend", choices=["hf", "unsloth"], default="hf")
    p.add_argument("--out", default="results/llm/local")
    p.add_argument("--log", default=None, help="also write all output to this UTF-8 file")
    p.add_argument("--label", default=None, help="policy name in the results (default llm-base / llm-grpo)")
    args = p.parse_args(argv)

    from training.logutil import tee_output, windows_compile_guard

    tee_output(args.log)
    windows_compile_guard()
    if args.backend == "unsloth":
        import unsloth  # noqa: F401  (import before transformers)
    policy = HFLocalPolicy(args.model, args.adapter, args.load_in_4bit, backend=args.backend)
    env = NexusgridEnvironment()
    records = []
    started = time.time()
    label = args.label or ("llm-grpo" if args.adapter else "llm-base")
    for task_id in [int(t) for t in args.tasks.split(",")]:
        for seed in range(EVAL_SEED_START, EVAL_SEED_START + args.seeds):
            before = dict(policy.stats())
            policy.reset(seed)
            obs = env.reset(seed=seed, task_id=task_id)
            while not obs.done:
                obs = env.step(policy.act(obs))
            summary = env.get_episode_summary()
            after = policy.stats()
            records.append({
                "policy": label, "task_id": task_id, "task_name": TASK_NAMES[task_id], "seed": seed,
                "split": "eval", "score": round(summary["score"], 6), "rubrics": summary["rubrics"],
                "steps": summary["ticks_used"], "invalid_actions": summary["invalid_actions"],
                "invalid_outputs": after["invalid_outputs"] - before["invalid_outputs"],
                "events": summary["events"],
            })
            print(f"[eval] task {task_id} seed {seed}: score {summary['score']:.3f} "
                  f"invalid outputs {records[-1]['invalid_outputs']}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    rows = summarise(records)
    (out / "summary.json").write_text(json.dumps({"model": args.model, "adapter": args.adapter, "rows": rows,
                                                  "policy_stats": policy.stats(),
                                                  "minutes": round((time.time() - started) / 60, 1)}, indent=2),
                                      encoding="utf-8")
    (out / "summary.md").write_text(markdown_table(rows) + "\n", encoding="utf-8")
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
