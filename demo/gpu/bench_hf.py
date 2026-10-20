"""A. HF transformers on CUDA: is the probe free in-graph?

Hand-written greedy decode loop over a StaticCache (so the decode step has
static shapes and can be CUDA-graphed by torch.compile). Prefill runs eager
(as HF recommends); only the 1-token decode step is compiled.

Modes (--mode):
  eager_none        eager decode step, no probe               (baseline for eager_*)
  eager_device      eager, probe = sigmoid(h_L @ w + b) on device, kept on device
  eager_cpu         eager, probe via h_L.cpu() + numpy every token (forces a sync)
  compile_none      torch.compile(mode="reduce-overhead") decode step, no probe
                                                              (baseline for compile_*)
  compile_ingraph   same, probe dot product INSIDE the compiled/CUDA-graphed step
  compile_cpu       same compiled step returns h_L; .cpu() + numpy per token
                    outside the graph
  correctness       HF reference p(true) on the check statements, fp32 + bf16
                    -> results/hf_reference.json (other engines compare to it)

h_L = hidden_states[L][:, -1] where L = probe layer from probe.npz
    = output of model.model.layers[L-1] (verified in export_probe.py).

    python bench_hf.py --mode all            # everything, appends results/results.csv
    python bench_hf.py --mode compile_ingraph
    python bench_hf.py --smoke --device cpu  # code-path check without a GPU
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, StaticCache

import common_gpu as C

MODES = ["eager_none", "eager_device", "eager_cpu",
         "compile_none", "compile_ingraph", "compile_cpu"]


class Decoder:
    def __init__(self, model, probe, mode: str, device: str, compile_backend: str | None):
        self.model, self.mode, self.device = model, mode, device
        self.L = probe["layer"]
        self.w = torch.from_numpy(probe["w"]).to(device)            # fp32 [hidden]
        self.b = torch.tensor(probe["b"], device=device, dtype=torch.float32)
        self.w_np, self.b_np = probe["w"], probe["b"]
        self.kind = mode.split("_")[1]                               # none|device|cpu|ingraph
        self.caches: dict[int, StaticCache] = {}
        self.probe_calls = 0        # probe evaluations in the last full-length generate()
        self._calls = 0
        self.sink = []                                               # on-device probe outputs
        step = self._step
        if mode.startswith("compile"):
            # mode="reduce-overhead" == inductor + CUDA graphs (cudagraph trees)
            kw = {"mode": "reduce-overhead"} if compile_backend is None else {"backend": compile_backend}
            step = torch.compile(step, fullgraph=True, dynamic=False, **kw)
        self.step = step

    # one decode step: [B,1] token ids -> (next ids [B,1], probe output or None)
    def _step(self, ids, cache):
        need_h = self.kind != "none"
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                         output_hidden_states=need_h, logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        if not need_h:
            return nxt, None
        h = out.hidden_states[self.L][:, -1]                        # [B, hidden]
        if self.kind in ("device", "ingraph"):
            return nxt, torch.sigmoid(h.float() @ self.w + self.b)   # [B]
        return nxt, h                                                # 'cpu': ship h out

    def _consume(self, pr):
        if pr is None:
            return
        self._calls += 1
        if self.kind == "cpu":
            hc = pr.float().cpu().numpy()                            # <-- sync every token
            _ = C.sigmoid(hc @ self.w_np + self.b_np)
        else:
            self.sink.append(pr.clone())                             # stays on device, async

    def cache(self, bs: int, max_len: int) -> StaticCache:
        if bs not in self.caches:
            self.caches[bs] = StaticCache(config=self.model.config, max_cache_len=max_len)
        c = self.caches[bs]
        c.reset()
        return c

    @torch.inference_mode()
    def generate(self, batch: list[list[int]], max_new: int) -> int:
        ids = torch.tensor(batch, device=self.device)
        bs, plen = ids.shape
        cache = self.cache(bs, plen + C.OUTPUT_LEN)
        self.sink.clear()
        self._calls = 0
        # prefill (eager, also produces token 1 and the prompt's last-token probe)
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                         output_hidden_states=self.kind != "none", logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        if self.kind != "none":
            h = out.hidden_states[self.L][:, -1]
            self._consume(h if self.kind == "cpu" else torch.sigmoid(h.float() @ self.w + self.b))
        toks = [nxt]
        for _ in range(max_new - 1):
            if self.mode.startswith("compile") and self.device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            nxt, pr = self.step(nxt, cache)
            self._consume(pr)
            nxt = nxt.clone()               # cudagraph outputs are overwritten next replay
            toks.append(nxt)
        if max_new == C.OUTPUT_LEN:
            self.probe_calls = self._calls
        return int(torch.cat(toks, 1).numel())


def load_model(device: str, dtype: torch.dtype):
    m = AutoModelForCausalLM.from_pretrained(C.MODEL_ID, dtype=dtype, attn_implementation="sdpa")
    return m.to(device).eval()


@torch.inference_mode()
def reference(device: str, probe, prompts) -> None:
    """HF p(true) on the check statements (raw text, last token -- as trained)."""
    res = {"layer": probe["layer"]}
    for name, dt in (("p_hf_fp32", torch.float32), ("p_hf_bf16", torch.bfloat16)):
        model = load_model(device, dt)
        w = torch.from_numpy(probe["w"]).to(device)
        ps, finals = [], []
        for ids in prompts["check_token_ids"]:
            out = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True)
            h = out.hidden_states[probe["layer"]][0, -1].float()
            ps.append(torch.sigmoid(h @ w + probe["b"]).item())
            # hidden_states[-1] is tied to last_hidden_state = POST final norm
            # (transformers utils/output_capturing.py tie_last_hidden_states)
            finals.append(out.hidden_states[-1][0, -1].float().cpu().tolist())
        res[name] = ps
        res[name.replace("p_hf", "final_postnorm_hf")] = finals
        # sanity: forward hook on layers[L-1] gives the same vector
        grab = {}
        hk = model.model.layers[probe["module_idx"]].register_forward_hook(
            lambda m, i, o: grab.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        out = model(input_ids=torch.tensor([prompts["check_token_ids"][0]], device=device),
                    output_hidden_states=True)
        hk.remove()
        assert torch.equal(grab["h"], out.hidden_states[probe["layer"]]), "layer mapping"
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.HF_REF_JSON.write_text(json.dumps(res, indent=1))
    print(f"[hf] reference -> {C.HF_REF_JSON}")
    print("  laptop fp32:", " ".join(f"{x:.3f}" for x in probe["ref_laptop"]))
    print("  hf    fp32 :", " ".join(f"{x:.3f}" for x in res["p_hf_fp32"]))
    print("  hf    bf16 :", " ".join(f"{x:.3f}" for x in res["p_hf_bf16"]))
    C.write_correctness("hf", "output_hidden_states_bf16", res["p_hf_bf16"], prompts, probe)
    C.write_correctness("hf", "output_hidden_states_fp32", res["p_hf_fp32"], prompts, probe)


def check_decode_matches_generate(model, dec: Decoder, prompt: list[int], device: str, n: int = 16):
    """Our StaticCache loop must reproduce HF generate() greedy tokens."""
    ids = torch.tensor([prompt], device=device)
    ref = model.generate(ids, max_new_tokens=n, do_sample=False, min_new_tokens=n,
                         pad_token_id=0)[0, len(prompt):].tolist()
    saved = C.OUTPUT_LEN
    C.OUTPUT_LEN = n
    dec.caches.clear()
    with torch.inference_mode():
        x = torch.tensor([prompt], device=device)
        cache = dec.cache(1, len(prompt) + n)
        out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        got = [int(nxt)]
        for _ in range(n - 1):
            if dec.mode.startswith("compile") and device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            nxt, _ = dec.step(nxt, cache)
            nxt = nxt.clone()
            got.append(int(nxt))
    dec.caches.clear()
    C.OUTPUT_LEN = saved
    agree = sum(a == b for a, b in zip(ref, got))
    print(f"  [check] {dec.mode}: static-cache loop vs generate(): {agree}/{n} tokens agree")
    if agree < n:
        # bf16 + different kernels (sdpa vs cudagraph'd inductor) can flip a
        # near-tie argmax late in the sequence; the first tokens must agree
        assert got[:4] == ref[:4], (got, ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", help=",".join(MODES + ["correctness", "all"]))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="no-GPU code-path check: fp32, tiny lengths, compile backend=aot_eager")
    args = ap.parse_args()

    if args.smoke:
        C.OUTPUT_LEN, C.BATCH_SIZES, C.REPEATS, C.WARMUP = 8, [1, 2], 1, 1
    probe, prompts = C.load_probe(), C.load_prompts()
    modes = MODES + ["correctness"] if args.mode == "all" else args.mode.split(",")
    torch._dynamo.config.cache_size_limit = 64
    torch.backends.cuda.matmul.allow_tf32 = True

    if "correctness" in modes:
        reference(args.device, probe, prompts)
    modes = [m for m in modes if m != "correctness"]
    if not modes:
        return

    dtype = torch.float32 if args.smoke else torch.bfloat16
    model = load_model(args.device, dtype)
    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)
    for mode in modes:
        print(f"\n[hf] mode={mode}  probe layer={probe['layer']} "
              f"(output of model.layers[{probe['module_idx']}])", flush=True)
        torch._dynamo.reset()
        dec = Decoder(model, probe, mode, args.device,
                      compile_backend="aot_eager" if args.smoke else None)
        check_decode_matches_generate(model, dec, prompts["bench_prompt_token_ids"][0], args.device)
        summ = C.bench(dec.generate, prompts["bench_prompt_token_ids"], f"hf/{mode}", sync)
        if dec.sink:     # make sure the on-device probe values are real
            assert torch.isfinite(torch.stack(dec.sink)).all()
        notes = ("prefill eager; decode step " +
                 ("torch.compile(mode='reduce-overhead', fullgraph=True)" if mode.startswith("compile")
                  else "eager") + "; StaticCache; sdpa")
        if args.smoke:
            notes = "SMOKE (not a measurement): " + notes
        C.write_results("hf" + ("-smoke" if args.smoke else ""),
                        f"transformers {transformers.__version__}",
                        mode, dec.kind, summ, prompts["prompt_len"], notes,
                        probe_calls=dec.probe_calls)


if __name__ == "__main__":
    main()
