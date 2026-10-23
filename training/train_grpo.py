#!/usr/bin/env python3
"""
GRPO fine-tuning of a language-model grid operator (LoRA, single GPU).

The reward is computed by the NexusGrid simulator for the model's *own*
action (see ``training/grpo_common.py``); nothing is replaced by a scripted
fallback. Prompts come from expert trajectories with exploration on training
seeds (< 9000); evaluate afterwards on held-out seeds with
``training/eval_llm_local.py``.

Tested setups
    * RTX 4060 Laptop (8 GB), WSL2 Ubuntu, Qwen3-1.7B in 4-bit + LoRA r=16:
          python training/train_grpo.py --model Qwen/Qwen3-1.7B --load-in-4bit --max-steps 300
    * CPU smoke test with a tiny local model (CI):
          python training/train_grpo.py --model <path> --max-steps 2 --dataset-episodes 4 --cpu

Hugging Face Hub upload is optional and only happens with ``--push-to-hub
user/repo``; the token is read from the ``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--backend", choices=["hf", "unsloth"], default="hf",
                   help="hf: transformers+peft+bitsandbytes; unsloth: faster kernels if unsloth is installed")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--tasks", default="1,2,3,4,5")
    p.add_argument("--dataset-episodes", type=int, default=250)
    p.add_argument("--epsilon", type=float, default=0.25, help="exploration rate while collecting prompt states")
    p.add_argument("--horizon", type=int, default=2, help="passive ticks after the model's action in the reward")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-generations", type=int, default=6)
    p.add_argument("--prompts-per-step", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 = no reference model, saves memory)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=96)
    p.add_argument("--use-vllm", action="store_true", help="generate with vLLM (Linux, enough VRAM)")
    p.add_argument("--cpu", action="store_true", help="CPU smoke test (fp32, no quantisation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/grpo/qwen3-1.7b")
    p.add_argument("--push-to-hub", default=None, help="optional user/repo for the LoRA adapter (needs HF_TOKEN)")
    p.add_argument("--log", default=None, help="also write all output to this UTF-8 file")
    return p.parse_args(argv)


class _Availability(tuple):
    """(available, version) tuple whose truth value is ``available``."""

    def __bool__(self) -> bool:
        return bool(self[0])


def trl_compat() -> None:
    """
    TRL 0.24 treats ``_is_package_available(name)`` as a bool, while transformers 5.x returns an
    ``(available, version)`` tuple, which is always truthy - TRL then imports optional packages such
    as mergekit that are not installed. Returning a tuple whose truth value is the first element
    satisfies both libraries. Harmless when the versions already agree.
    """
    import transformers.utils.import_utils as iu

    original = iu._is_package_available
    if getattr(original, "_nexusgrid_patched", False):
        return

    def patched(pkg_name, return_version=False):
        result = original(pkg_name, return_version=return_version)
        return _Availability(result) if isinstance(result, tuple) else result

    patched._nexusgrid_patched = True
    iu._is_package_available = patched


def dtype_kwargs(torch_dtype) -> dict:
    """transformers >= 4.56 renamed ``torch_dtype`` to ``dtype``."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": torch_dtype} if (major, minor) >= (4, 56) else {"torch_dtype": torch_dtype}


def build_dataset(args, tokenizer):
    from datasets import Dataset

    from training.grpo_common import collect_states

    tasks = [int(t) for t in args.tasks.split(",")]
    records = collect_states(tasks, args.dataset_episodes, epsilon=args.epsilon, seed=args.seed)
    rows = []
    for r in records:
        prompt = tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        rows.append({"prompt": prompt, "task_id": r["task_id"], "seed": r["seed"], "prefix": r["prefix"],
                     "expert_action": r["expert_action"]})
    return Dataset.from_list(rows).shuffle(seed=args.seed)


def load_model(args):
    import torch

    if args.backend == "unsloth":
        from unsloth import FastLanguageModel  # noqa: F401  (import before trl so its patches apply)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_prompt_length + args.max_completion_length,
            load_in_4bit=args.load_in_4bit, fast_inference=args.use_vllm, max_lora_rank=args.lora_r,
            gpu_memory_utilization=0.55,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )
        return model, tokenizer, None

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {}
    if args.cpu:
        kwargs.update(dtype_kwargs(torch.float32))
    else:
        kwargs.update(dtype_kwargs(torch.bfloat16))
        kwargs["device_map"] = {"": 0}
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.load_in_4bit and not args.cpu:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules="all-linear")
    return model, tokenizer, peft_config


def main(argv=None) -> int:
    args = parse_args(argv)
    from training.logutil import tee_output

    tee_output(args.log)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    trl_compat()
    model, tokenizer, peft_config = load_model(args)
    from trl import GRPOConfig, GRPOTrainer

    from training.grpo_common import make_reward_fn

    started = time.time()
    dataset = build_dataset(args, tokenizer)
    print(f"[grpo] {len(dataset)} prompts built in {time.time() - started:.1f}s", flush=True)

    batch = args.num_generations * args.prompts_per_step
    config_kwargs = dict(
        output_dir=str(out / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        temperature=args.temperature,
        beta=args.beta,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        seed=args.seed,
        use_vllm=args.use_vllm and args.backend == "hf",
        bf16=not args.cpu,
        use_cpu=args.cpu,
        gradient_checkpointing=not args.cpu and args.backend == "hf",
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * args.max_steps)),
    )
    import inspect

    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    if "max_prompt_length" in accepted:
        config_kwargs["max_prompt_length"] = args.max_prompt_length
    config = GRPOConfig(**{k: v for k, v in config_kwargs.items() if k in accepted})
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}  # TRL 0.24 writes to this attribute; transformers 5.x dropped it
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(args.horizon)],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()

    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    history = trainer.state.log_history
    (out / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"minutes": round((time.time() - started) / 60, 1), "steps": args.max_steps,
               "prompts": len(dataset), "adapter": str(adapter_dir)}
    rewards = [h.get("reward") for h in history if h.get("reward") is not None]
    if rewards:
        k = max(1, len(rewards) // 10)
        summary["reward_first_10pct"] = sum(rewards[:k]) / k
        summary["reward_last_10pct"] = sum(rewards[-k:]) / k
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)

    try:
        from training.plots import plot_grpo_log

        plot_grpo_log(out / "train_log.json", out / "reward_curve.png")
    except Exception as exc:  # plotting is optional
        print(f"[grpo] plot skipped: {exc}", flush=True)

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[grpo] --push-to-hub given but HF_TOKEN is not set; skipping upload", flush=True)
        else:
            trainer.model.push_to_hub(args.push_to_hub, token=token, private=True)
            tokenizer.push_to_hub(args.push_to_hub, token=token, private=True)
            print(f"[grpo] adapter pushed to {args.push_to_hub} (private)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
