"""Benchmark the three laptop-measurable costs of running a probe alongside
generation. Produces a clean table for a slide.

The claims this validates (all measurable on MPS, no GPU/vLLM needed):

  1. hidden-states overhead  -- output_hidden_states=True adds ~nothing
                                (the activations are already computed).
  2. probe compute           -- the probe is a single matmul; negligible
                                vs one model forward step.
  3. on-device vs .cpu()     -- the headline. Applying the probe on-device
                                is ~free; doing `h.cpu().numpy()` per token
                                forces a device sync each step and serializes
                                the decode loop. Same math, very different cost.

Timing notes: MPS (like CUDA) is async, so we torch.mps.synchronize() around
every measured region and warm up before timing. Numbers are medians.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import torch

from common import PROBE_PATH, load_model_and_tokenizer

REPEATS = 30          # for the fast micro-benchmarks
WARMUP = 5
DECODE_TOKENS = 48    # realistic generation length for the decode-loop bench
DECODE_REPEATS = 3
PROMPT = "The capital of Australia is"


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def median_time(fn, repeats: int, warmup: int, device: str) -> float:
    """Median seconds per call. fn should do exactly one unit of work, big
    enough that a single trailing sync dominates its own launch overhead
    (forward passes, full decode loops)."""
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def amortized_time(loop_fn, k: int, repeats: int, warmup: int,
                   device: str) -> float:
    """Median seconds PER unit, where loop_fn does k units back-to-back with
    one trailing sync. This is the right way to price a tiny on-device op: in
    production you queue work and sync when you actually need the value, not
    after every call."""
    for _ in range(warmup):
        loop_fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        loop_fn()
        sync(device)
        samples.append((time.perf_counter() - t0) / k)
    return float(np.median(samples))


def fold_probe(bundle):
    """Fold StandardScaler + LogisticRegression into one affine map.

        z = (x - mean) / scale
        logit = z . coef + intercept
              = x . (coef/scale) + (intercept - sum(mean*coef/scale))
    """
    pipe = bundle["probe"]
    sc = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    coef = lr.coef_[0]                       # [hidden]
    w = coef / sc.scale_                      # [hidden]
    b = float(lr.intercept_[0] - np.sum(sc.mean_ * coef / sc.scale_))
    return w.astype(np.float32), b, pipe


def main() -> None:
    bundle = joblib.load(PROBE_PATH)
    layer = bundle["layer"]
    w_np, b_np, pipe = fold_probe(bundle)

    model, tok, device = load_model_and_tokenizer()
    w_t = torch.from_numpy(w_np).to(device)          # [hidden]
    b_t = torch.tensor(b_np, device=device)

    # one real hidden vector to probe + correctness check ------------------
    enc = tok(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[layer][:, -1, :]           # [1, hidden], on device

    p_dev = torch.sigmoid(h @ w_t + b_t).item()
    p_sk = float(pipe.predict_proba(h.float().cpu().numpy())[0, 1])
    print(f"\n[check] folded on-device p(true)={p_dev:.6f}  "
          f"sklearn p(true)={p_sk:.6f}  |Δ|={abs(p_dev - p_sk):.2e}")

    # =====================================================================
    # 1. hidden-states overhead: forward pass, HS off vs on
    # =====================================================================
    def fwd(hs: bool):
        with torch.no_grad():
            model(**enc, output_hidden_states=hs, use_cache=False)

    t_hs_off = median_time(lambda: fwd(False), REPEATS, WARMUP, device)
    t_hs_on = median_time(lambda: fwd(True), REPEATS, WARMUP, device)

    # =====================================================================
    # 2. probe compute, isolated: on-device matmul vs naive .cpu()+numpy
    #    same math; difference is where it runs + the forced per-call sync.
    #    on-device is queued (sync once); .cpu() must sync every call.
    # =====================================================================
    K_DEV, K_CPU = 2000, 1000

    def loop_on_device():
        s = b_t
        for _ in range(K_DEV):                        # queued, no inner sync
            s = s + torch.sigmoid(h @ w_t + b_t).squeeze()

    def loop_via_cpu():
        for _ in range(K_CPU):
            hc = h.cpu().numpy()                      # <-- forces a sync each call
            _ = 1.0 / (1.0 + np.exp(-(hc @ w_np + b_np)))

    t_probe_dev = amortized_time(loop_on_device, K_DEV, 5, 2, device)
    t_probe_cpu = amortized_time(loop_via_cpu, K_CPU, 5, 2, device)

    # =====================================================================
    # 3. realistic decode loop: model-only vs +on-device probe vs +cpu probe
    # =====================================================================
    def decode(mode: str) -> None:
        with torch.no_grad():
            cur = tok(PROMPT, return_tensors="pt").to(device)["input_ids"]
            past = None
            for _ in range(DECODE_TOKENS):
                o = model(input_ids=cur, past_key_values=past,
                          use_cache=True, output_hidden_states=True)
                past = o.past_key_values
                hh = o.hidden_states[layer][:, -1, :]
                if mode == "device":
                    _ = torch.sigmoid(hh @ w_t + b_t)
                elif mode == "cpu":
                    hc = hh.cpu().numpy()
                    _ = 1.0 / (1.0 + np.exp(-(hc @ w_np + b_np)))
                cur = o.logits[:, -1, :].argmax(-1, keepdim=True)

    t_base = median_time(lambda: decode("none"), DECODE_REPEATS, 1, device)
    t_dev = median_time(lambda: decode("device"), DECODE_REPEATS, 1, device)
    t_cpu = median_time(lambda: decode("cpu"), DECODE_REPEATS, 1, device)

    # ---- report ---------------------------------------------------------
    us = 1e6
    ms = 1e3
    print("\n" + "=" * 64)
    print(f"  PROBE COST BENCHMARK  ({device}, {model.config._name_or_path})")
    print(f"  probe @ layer {layer},  hidden_size={model.config.hidden_size}")
    print("=" * 64)

    print("\n  1. HIDDEN-STATES OVERHEAD  (one full forward pass)")
    print(f"     output_hidden_states=False : {t_hs_off * ms:8.2f} ms")
    print(f"     output_hidden_states=True  : {t_hs_on * ms:8.2f} ms")
    print(f"     -> overhead                : {(t_hs_on - t_hs_off) * ms:+8.2f} ms"
          f"  ({(t_hs_on / t_hs_off - 1) * 100:+.1f}%)")

    print("\n  2. PROBE COMPUTE  (per call, isolated)")
    print(f"     on-device matmul           : {t_probe_dev * us:8.2f} us")
    print(f"     via .cpu().numpy()         : {t_probe_cpu * us:8.2f} us")
    print(f"     -> .cpu() is               : {t_probe_cpu / t_probe_dev:8.1f}x slower")

    print(f"\n  3. DECODE LOOP  ({DECODE_TOKENS} tokens, probe every step)")
    print(f"     model only                 : {t_base * ms:8.2f} ms"
          f"   ({DECODE_TOKENS / t_base:6.1f} tok/s)")
    print(f"     + on-device probe          : {t_dev * ms:8.2f} ms"
          f"   ({DECODE_TOKENS / t_dev:6.1f} tok/s)")
    print(f"     + .cpu() probe per token   : {t_cpu * ms:8.2f} ms"
          f"   ({DECODE_TOKENS / t_cpu:6.1f} tok/s)")
    add_dev = (t_dev - t_base) / DECODE_TOKENS * ms
    add_cpu = (t_cpu - t_base) / DECODE_TOKENS * ms
    print(f"     -> added latency / token   :  on-device {add_dev:+.3f} ms"
          f"   vs   .cpu() {add_cpu:+.3f} ms")
    print(f"     -> on-device probe overhead: {(t_dev / t_base - 1) * 100:+.1f}%"
          f"   |   .cpu() probe overhead: {(t_cpu / t_base - 1) * 100:+.1f}%")
    print("=" * 64)
    print("  takeaway: the probe is free; reading activations OFF-DEVICE is the bill.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
