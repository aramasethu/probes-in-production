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
every measured region and warm up before timing. Every number is a median with
its IQR; every overhead % comes with a 95% bootstrap CI (resampling the timing
samples of both arms). The three decode variants are run interleaved
(round-robin) so thermal / background drift hits all of them equally. An
overhead whose CI spans 0 is "within noise", not a measured saving or cost.
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
DECODE_REPEATS = 20   # per decode variant, interleaved
N_BOOT = 5000
PROMPT = "The capital of Australia is"


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device: str) -> float:
    t0 = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - t0


def time_samples(fn, repeats: int, warmup: int, device: str) -> np.ndarray:
    """Seconds per call, one sample per repeat. fn should do exactly one unit
    of work, big enough that a single trailing sync dominates its own launch
    overhead (forward passes, full decode loops)."""
    for _ in range(warmup):
        fn()
    sync(device)
    return np.array([timed(fn, device) for _ in range(repeats)])


def interleaved_samples(fns: dict, repeats: int, warmup: int,
                        device: str) -> dict[str, np.ndarray]:
    """Like time_samples for several variants, run round-robin so slow drift
    (thermals, background load) is shared across variants, not confounded."""
    for _ in range(warmup):
        for fn in fns.values():
            fn()
    sync(device)
    out = {k: [] for k in fns}
    for _ in range(repeats):
        for k, fn in fns.items():
            out[k].append(timed(fn, device))
    return {k: np.array(v) for k, v in out.items()}


def summ(x: np.ndarray) -> tuple[float, float, float]:
    """(median, q25, q75)"""
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return float(med), float(q25), float(q75)


def overhead_ci(base: np.ndarray, var: np.ndarray, seed: int = 0):
    """Overhead % = median(var)/median(base) - 1, with a 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    point = np.median(var) / np.median(base) - 1
    bs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        bs[i] = (np.median(rng.choice(var, len(var))) /
                 np.median(rng.choice(base, len(base))) - 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point * 100, lo * 100, hi * 100


def fmt_ci(pt: float, lo: float, hi: float) -> str:
    tag = "  within noise" if lo <= 0 <= hi else ""
    return f"{pt:+6.1f}%  [95% CI {lo:+.1f}, {hi:+.1f}]{tag}"


def amortized_samples(loop_fn, k: int, repeats: int, warmup: int,
                      device: str) -> np.ndarray:
    """Seconds PER unit (one sample per repeat), where loop_fn does k units
    back-to-back with one trailing sync. This is the right way to price a tiny
    on-device op: in production you queue work and sync when you actually need
    the value, not after every call."""
    return time_samples(loop_fn, repeats, warmup, device) / k


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

    hs = interleaved_samples({"off": lambda: fwd(False),
                              "on": lambda: fwd(True)}, REPEATS, WARMUP, device)

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

    s_probe_dev = amortized_samples(loop_on_device, K_DEV, 15, 2, device)
    s_probe_cpu = amortized_samples(loop_via_cpu, K_CPU, 15, 2, device)

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

    dec = interleaved_samples({m: (lambda m=m: decode(m))
                               for m in ("none", "device", "cpu")},
                              DECODE_REPEATS, 2, device)

    # ---- report ----------------------------------------------------------
    us = 1e6
    ms = 1e3

    def row(label, x, scale, unit, extra=""):
        med, q25, q75 = summ(x)
        print(f"     {label:27s}: {med * scale:8.2f} {unit}  "
              f"(IQR {q25 * scale:.2f}-{q75 * scale:.2f}){extra}")

    print("\n" + "=" * 72)
    print(f"  PROBE COST BENCHMARK  ({device}, {model.config._name_or_path})")
    print(f"  probe @ layer {layer},  hidden_size={model.config.hidden_size}")
    print("  medians with IQR; overhead % with 95% bootstrap CI")
    print("=" * 72)

    print(f"\n  1. HIDDEN-STATES OVERHEAD  (one full forward pass, n={REPEATS} each)")
    row("output_hidden_states=False", hs["off"], ms, "ms")
    row("output_hidden_states=True", hs["on"], ms, "ms")
    print(f"     -> overhead                : {fmt_ci(*overhead_ci(hs['off'], hs['on']))}")

    print(f"\n  2. PROBE COMPUTE  (per call, isolated, n={len(s_probe_dev)} "
          f"batches of {K_DEV}/{K_CPU})")
    row("on-device matmul", s_probe_dev, us, "us")
    row("via .cpu().numpy()", s_probe_cpu, us, "us")
    r_pt, r_lo, r_hi = overhead_ci(s_probe_dev, s_probe_cpu)
    print(f"     -> .cpu() is               : {1 + r_pt / 100:8.1f}x slower  "
          f"[95% CI {1 + r_lo / 100:.1f}x, {1 + r_hi / 100:.1f}x]")

    print(f"\n  3. DECODE LOOP  ({DECODE_TOKENS} tokens, probe every step, "
          f"n={DECODE_REPEATS} per variant, interleaved)")
    for label, k in (("model only", "none"), ("+ on-device probe", "device"),
                     ("+ .cpu() probe per token", "cpu")):
        row(label, dec[k], ms, "ms",
            f"   ({DECODE_TOKENS / np.median(dec[k]):6.1f} tok/s)")
    t_base, t_dev, t_cpu = (np.median(dec[k]) for k in ("none", "device", "cpu"))
    add_dev = (t_dev - t_base) / DECODE_TOKENS * ms
    add_cpu = (t_cpu - t_base) / DECODE_TOKENS * ms
    print(f"     -> added latency / token   :  on-device {add_dev:+.3f} ms"
          f"   vs   .cpu() {add_cpu:+.3f} ms")
    print(f"     -> on-device probe overhead: "
          f"{fmt_ci(*overhead_ci(dec['none'], dec['device']))}")
    print(f"     -> .cpu() probe overhead   : "
          f"{fmt_ci(*overhead_ci(dec['none'], dec['cpu']))}")
    print("=" * 72)
    print("  takeaway: the probe is free; reading activations OFF-DEVICE is the bill.")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
