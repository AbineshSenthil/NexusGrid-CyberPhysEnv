#!/usr/bin/env python3
"""
GRPO fine-tuning of a language-model grid operator (LoRA, single GPU).

Two phases, both on training seeds (< 9000):

1. Supervised warm start (``--sft-episodes``, 0 to skip): the LoRA adapter learns the expert's action
   in states visited by the expert with exploration. The adapter is saved as ``sft_adapter`` so it can be
   evaluated on its own.
2. GRPO: the reward is computed by the NexusGrid simulator for the model's *own* sampled action (see
   ``training/grpo_common.py``); nothing is replaced by a scripted fallback.

Evaluate afterwards on held-out seeds with ``training/eval_llm_local.py``.

Tested setups
    * RTX 4060 Laptop GPU (8 GB), Windows 11, Unsloth, Qwen3-1.7B in 4-bit + LoRA r=16 (about an hour):
          python training/train_grpo.py --backend unsloth --model unsloth/Qwen3-1.7B-unsloth-bnb-4bit --load-in-4bit
    * CPU smoke test with a tiny local model (training/make_tiny_model.py):
          python training/train_grpo.py --model <path> --cpu --max-steps 2 --dataset-episodes 4 --sft-episodes 4

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
    p.add_argument("--min-gap", type=float, default=0.01,
                   help="keep states where the expert's action beats waiting by this much reward (0 = keep all)")
    p.add_argument("--horizon", type=int, default=2, help="passive ticks after the model's action in the reward")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4, help="micro-batches per optimizer step")
    p.add_argument("--micro-batch", type=int, default=4,
                   help="sequences per forward/backward pass (lower = less VRAM); must divide the generation batch")
    p.add_argument("--sft-episodes", type=int, default=200,
                   help="expert episodes for the supervised warm start (0 = GRPO only)")
    p.add_argument("--sft-epochs", type=int, default=1)
    p.add_argument("--sft-lr", type=float, default=1e-4)
    p.add_argument("--sft-grad-accum", type=int, default=8)
    p.add_argument("--vram-fraction", type=float, default=0.9,
                   help="cap on the GPU memory PyTorch may use, so Windows cannot silently spill into shared memory")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 = no reference model, saves memory)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-prompt-length", type=int, default=1792)
    p.add_argument("--max-completion-length", type=int, default=64)
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
    records = collect_states(tasks, args.dataset_episodes, epsilon=args.epsilon, seed=args.seed,
                             min_gap=args.min_gap, horizon=args.horizon)
    counts = {t: sum(1 for r in records if r["task_id"] == t) for t in tasks}
    print(f"[grpo] prompts per task {counts} (min gap {args.min_gap})", flush=True)
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
            gpu_memory_utilization=0.55, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )
        # Pre-quantised checkpoints can declare float16 in their config while the weights were loaded in
        # bfloat16; Unsloth derives its autocast dtype from the config, so make the config tell the truth.
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        for attr in ("dtype", "torch_dtype"):
            try:
                setattr(model.config, attr, compute_dtype)
            except Exception:
                pass
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


def sft_examples(records, tokenizer, max_len: int):
    """(input_ids, labels) pairs: the chat prompt is masked, the expert's action JSON is the target."""
    eos = tokenizer.eos_token or ""
    examples = []
    for r in records:
        prompt = tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        target = json.dumps(json.loads(r["expert_action"]), separators=(",", ":")) + eos
        p_ids = tokenizer(text=prompt, add_special_tokens=False)["input_ids"]
        t_ids = tokenizer(text=target, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[-max_len:]
        labels = ([-100] * len(p_ids) + t_ids)[-max_len:]
        examples.append((ids, labels))
    return examples


def sft_warm_start(model, tokenizer, args, out: Path) -> dict:
    """Behaviour cloning of the expert on training seeds: one LoRA, plain PyTorch loop, batch size 1."""
    import math
    import random

    import torch

    from training.grpo_common import collect_states

    tasks = [int(t) for t in args.tasks.split(",")]
    started = time.time()
    records = collect_states(tasks, args.sft_episodes, epsilon=args.epsilon, seed=args.seed + 1)
    examples = sft_examples(records, tokenizer, args.max_prompt_length + args.max_completion_length)
    print(f"[sft] {len(examples)} expert examples from {args.sft_episodes} episodes "
          f"in {time.time() - started:.1f}s", flush=True)

    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.sft_lr, weight_decay=0.0)
    total = max(1, math.ceil(args.sft_epochs * len(examples) / args.sft_grad_accum))
    warmup = max(1, int(0.05 * total))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_autocast and torch.cuda.is_bf16_supported() else torch.float16
    rng = random.Random(args.seed)
    model.train()
    history, window, step = [], [], 0
    for epoch in range(args.sft_epochs):
        rng.shuffle(examples)
        for i, (ids, labels) in enumerate(examples):
            input_ids = torch.tensor([ids], device=device)
            label_ids = torch.tensor([labels], device=device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                loss = model(input_ids=input_ids, labels=label_ids).loss
            (loss / args.sft_grad_accum).backward()
            window.append(float(loss.detach()))
            last = i == len(examples) - 1
            if (i + 1) % args.sft_grad_accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total or last:
                    mean = sum(window) / len(window)
                    history.append({"step": step, "epoch": epoch, "loss": round(mean, 4)})
                    print(f"[sft] step {step}/{total} loss {mean:.4f} ({time.time() - started:.0f}s)", flush=True)
                    window = []
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler
    if use_autocast:
        torch.cuda.empty_cache()
    sft_dir = out / "sft_adapter"
    model.save_pretrained(str(sft_dir))
    tokenizer.save_pretrained(str(sft_dir))
    (out / "sft_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"examples": len(examples), "episodes": args.sft_episodes, "epochs": args.sft_epochs,
               "optimizer_steps": step, "loss_first": history[0]["loss"] if history else None,
               "loss_last": history[-1]["loss"] if history else None,
               "minutes": round((time.time() - started) / 60, 1), "adapter": str(sft_dir)}
    print(f"[sft] done: {json.dumps(summary)}", flush=True)
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    from training.logutil import tee_output, windows_compile_guard

    tee_output(args.log)
    windows_compile_guard()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Reclaim cached blocks early and avoid fragmentation; with the per-process cap below, PyTorch frees
    # its cache instead of growing into Windows' shared-memory fallback (which slows training ~10x).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.7,max_split_size_mb:256")
    if args.backend == "unsloth":
        import unsloth  # noqa: F401  (must come before transformers/trl so its patches apply)
    trl_compat()
    model, tokenizer, peft_config = load_model(args)
    import torch

    if torch.cuda.is_available() and not args.cpu and 0 < args.vram_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    sft_summary = None
    if args.sft_episodes > 0:
        if peft_config is not None:  # hf backend: wrap now so the warm start and GRPO train the same LoRA
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)
            peft_config = None
        sft_summary = sft_warm_start(model, tokenizer, args, out)
    from trl import GRPOConfig, GRPOTrainer

    from training.grpo_common import make_reward_fn

    started = time.time()
    dataset = build_dataset(args, tokenizer)
    print(f"[grpo] {len(dataset)} prompts built in {time.time() - started:.1f}s", flush=True)

    bf16_ok = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    generation_batch = args.num_generations * args.prompts_per_step
    config_kwargs = dict(
        output_dir=str(out / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=min(args.micro_batch, generation_batch),
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        generation_batch_size=generation_batch,
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
        bf16=not args.cpu and bf16_ok,
        fp16=not args.cpu and not bf16_ok,
        use_cpu=args.cpu,
        gradient_checkpointing=not args.cpu and args.backend == "hf",
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * args.max_steps)),
    )
    if args.backend == "unsloth":
        # Unsloth picks the mixed-precision mode from the model dtype only when both flags are off;
        # setting bf16=True ourselves leaves its autocast at fp16 and breaks the LoRA kernels.
        config_kwargs["bf16"] = False
        config_kwargs["fp16"] = False
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16" if bf16_ok else "fp16"

    import inspect

    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    if "max_prompt_length" in accepted:
        config_kwargs["max_prompt_length"] = args.max_prompt_length
    config = GRPOConfig(**{k: v for k, v in config_kwargs.items() if k in accepted})
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}  # TRL 0.24 writes to this attribute; transformers 5.x dropped it
    from training.logutil import quiet_transformers

    quiet_transformers()
    callbacks = []
    if torch.cuda.is_available() and not args.cpu:
        from transformers import TrainerCallback

        class FreeCudaCache(TrainerCallback):
            """Return cached blocks after every step so prompt-length variation cannot fragment VRAM."""

            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.empty_cache()

        callbacks.append(FreeCudaCache())
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(args.horizon)],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    if args.backend == "unsloth" and torch.cuda.is_available():
        # Unsloth's GRPO loss reads its autocast dtype from an environment variable that
        # transformers 5.x no longer sets; pin it to the dtype the LoRA weights actually use.
        trainer._autocast_dtype = torch.bfloat16 if bf16_ok else torch.float16
    trainer.train()

    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    history = trainer.state.log_history
    (out / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"minutes": round((time.time() - started) / 60, 1), "steps": args.max_steps, "sft": sft_summary,
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
