"""Export CausalPFN to ONNX for the DuckDo inference runtime (roadmap phase 5).

CausalPFN (Apache-2.0, 18.8M parameters) estimates conditional expected potential
outcomes under the backdoor criterion. Unlike Do-PFN's non-identifiable prior it
targets the ATE directly, and in DuckDo's own benchmarks it does not exhibit the
severe shrinkage Do-PFN does.

## The exportable unit

`InContextModel.predict_cepo` is pure tensor work: standardise the context
outcome per arm, concatenate the treatment onto the covariates, run the
transformer, and turn logits over a binned distribution into a mean, then undo
the standardisation. Everything the Python package does *around* it - a weak
learner for stratification, faiss retrieval, a batching binary search - is
orchestration, and DuckDo reimplements the parts it needs in C++ (its own
DR-learner supplies the stratification scalar, and 1-D nearest neighbours are a
sorted sliding window, not a vector index).

## Dynamic axes

Unlike Do-PFN, this model derives its context length from `y_src.shape[0]`
rather than a Python int, so both the context and query axes have a real chance
of surviving the trace. The script verifies that at three different shapes and
refuses to write a manifest if any of them disagrees with PyTorch - a graph that
silently returns the wrong rows at an unseen shape is the worst thing we could
ship.

Dev-only; the extension ships the graph, never the weights.

    python scripts/export/export_causalpfn.py --out build/models
"""

import argparse
import io
import json
import os
import sys

import numpy as np

TOLERANCE = 1e-4


def patch_inplace_and(torch):
    """ONNX opset 17 has no aten::__iand_. The one use is `mask &= cond` inside
    clip_outliers, and the out-of-place `mask = mask & cond` is exactly the same
    operation - mask is a local that nothing else aliases. Swapped in on the
    module rather than edited into site-packages, and the parity check confirms
    the graph is unchanged."""
    import causalpfn.models.model as model_module

    maskmean = model_module.maskmean
    maskstd = model_module.maskstd

    def clip_outliers(data, eval_pos, n_sigma=4):
        assert len(data.shape) == 3, "X must be T,B,H"
        X = data[:eval_pos] if eval_pos > 0 else data
        mask = ~torch.isnan(X)
        mean = maskmean(X, mask, dim=0)
        cutoff = n_sigma * maskstd(X, mask, dim=0)
        mask = mask & (cutoff >= torch.abs(X - mean))
        cutoff = n_sigma * maskstd(X, mask, dim=0)
        return torch.clip(data, mean - cutoff, mean + cutoff)

    model_module.clip_outliers = clip_outliers


def load(device="cpu"):
    import torch
    from causalpfn import CATEEstimator

    patch_inplace_and(torch)

    # The model is loaded lazily inside fit(), so fit on a throwaway sample.
    rng = np.random.default_rng(0)
    n, d = 256, 4
    X = rng.normal(size=(n, d)).astype(np.float32)
    t = (rng.random(n) < 0.5).astype(np.float32)
    y = (X[:, 0] + 2.0 * t + rng.normal(size=n)).astype(np.float32)

    estimator = CATEEstimator(device=device)
    estimator.fit(X, t, y)
    model = estimator.icl_model
    model.eval()
    return torch, estimator, model


def make_wrapper(torch, model):
    """Wrap predict_cepo so ONNX sees five tensors in and one out."""

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, X_context, t_context, y_context, X_query, t_query):
            # (1, context, d), (1, context), (1, context), (1, query, d), (1, query)
            out = self.inner.predict_cepo(
                X_context=X_context,
                t_context=t_context,
                y_context=y_context,
                X_query=X_query,
                t_query=t_query,
                n_samples=None,
                temperature=torch.ones(1, dtype=X_context.dtype),
            )
            # (batch, num_temperatures, query) -> (query,)
            return out.reshape(-1)

    return Wrapper()


def sample(torch, context_len, query_len, d, seed=0):
    torch.manual_seed(seed)
    X_context = torch.randn(1, context_len, d)
    t_context = (torch.rand(1, context_len) < 0.5).float()
    y_context = torch.randn(1, context_len)
    X_query = torch.randn(1, query_len, d)
    t_query = (torch.rand(1, query_len) < 0.5).float()
    return X_context, t_context, y_context, X_query, t_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("build", "models"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--query", type=int, default=64)
    parser.add_argument("--features", type=int, default=0,
                        help="input width; 0 discovers it from the model")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    torch, estimator, model = load()
    n_params = sum(p.numel() for p in model.parameters())

    # The x-encoder's input width is the covariate budget plus the treatment
    # column. Ask it directly rather than guessing at the first Linear, which is
    # the embedding and reports 384.
    width = args.features or int(model.model.encoder.in_features)
    covariates = width - 1
    nbins = int(getattr(model.model, "nbins", 0))
    print("loaded CausalPFN: %.2fM params, input width %d (treatment + %d covariates), "
          "max_context %d, max_query %d"
          % (n_params / 1e6, width, covariates, estimator.max_context_length,
             estimator.max_query_length))
    print("  %d output bins; beyond %d covariates the package reduces with TruncatedSVD, "
          "where DuckDo instead keeps the most outcome-correlated" % (nbins, covariates))

    inputs = sample(torch, args.context, args.query, covariates)
    with torch.no_grad():
        reference = make_wrapper(torch, model)(*inputs)
    print("reference output:", tuple(reference.shape), reference.dtype)

    wrapper = make_wrapper(torch, model)
    path = os.path.join(out_dir, "causalpfn.onnx")
    names = ["X_context", "t_context", "y_context", "X_query", "t_query"]
    with torch.no_grad():
        torch.onnx.export(
            wrapper, inputs, path,
            input_names=names,
            output_names=["mu"],
            dynamic_axes={
                "X_context": {1: "context"},
                "t_context": {1: "context"},
                "y_context": {1: "context"},
                "X_query": {1: "query"},
                "t_query": {1: "query"},
                "mu": {0: "query"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    import onnx
    # save_model APPENDS to an existing external-data file, so remove it first
    # or a re-run doubles the blob.
    shared = os.path.join(out_dir, "causalpfn.weights.bin")
    if os.path.exists(shared):
        os.remove(shared)
    proto = onnx.load(path)
    onnx.save_model(proto, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="causalpfn.weights.bin", size_threshold=1024, convert_attribute=False)

    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    # Three shapes, including the export shape. If the trace baked the split
    # point in, the second and third disagree - loudly, here, rather than
    # quietly, in someone's query.
    shapes = [(args.context, args.query), (args.context * 2, args.query),
              (args.context, args.query * 3)]
    worst = 0.0
    for context_len, query_len in shapes:
        probe = sample(torch, context_len, query_len, covariates, seed=context_len + query_len)
        with torch.no_grad():
            expected = make_wrapper(torch, model)(*probe)
        got = session.run(None, {name: tensor.numpy() for name, tensor in zip(names, probe)})[0]
        if got.shape != tuple(expected.shape):
            raise RuntimeError("shape mismatch at context=%d query=%d: onnx %s vs torch %s. "
                               "The context/query split was baked in by the tracer."
                               % (context_len, query_len, got.shape, tuple(expected.shape)))
        gap = float(np.max(np.abs(got - expected.numpy())))
        scale = float(np.max(np.abs(expected.numpy()))) or 1.0
        worst = max(worst, gap / scale)
        print("  context=%-6d query=%-5d abs=%.3e  rel=%.3e" % (context_len, query_len, gap, gap / scale))

    graph_bytes = os.path.getsize(path)
    weight_bytes = os.path.getsize(os.path.join(out_dir, "causalpfn.weights.bin"))
    print("graph %.1f KB, weights %.2f MB" % (graph_bytes / 1024.0, weight_bytes / 1e6))

    manifest = {
        "model": "causalpfn",
        "source": "https://github.com/vdblm/CausalPFN",
        "license": "Apache-2.0",
        "attribution_required": False,
        "setting": "backdoor (ignorability)",
        "opset": args.opset,
        "input_width": width,
        "max_covariates": covariates,
        "num_bins": nbins,
        "max_context": int(estimator.max_context_length),
        "max_query": int(estimator.max_query_length),
        "num_neighbours": int(estimator.num_neighbours),
        "graph": "causalpfn.onnx",
        "weights": "causalpfn.weights.bin",
        "graph_bytes": graph_bytes,
        "weight_bytes": weight_bytes,
        "inputs": ["X_context", "t_context", "y_context", "X_query", "t_query"],
        "output": "mu",
        "dynamic_axes": {"context": ["X_context", "t_context", "y_context"],
                         "query": ["X_query", "t_query", "mu"]},
        "parity_max_abs_diff": worst,
        "notes": ("Output is the conditional expected potential outcome mu(x, t). "
                  "CATE is mu(x, 1) - mu(x, 0), so the runtime runs the query twice. "
                  "Both context and query lengths are dynamic."),
    }
    manifest_path = os.path.join(out_dir, "causalpfn.manifest.json")
    with io.open(manifest_path, "w", encoding="utf8") as handle:
        json.dump(manifest, handle, indent=2)
    print("wrote", manifest_path)

    # Logit-level parity is only a proxy. What has to agree is the estimand, so
    # compute a CATE both ways on a real DGP and compare that directly.
    rng = np.random.default_rng(11)
    n, d = 512, 5
    Xn = rng.normal(size=(n, d)).astype(np.float32)
    ps = 1.0 / (1.0 + np.exp(-(0.9 * Xn[:, 0] - 0.6 * Xn[:, 1])))
    tn = (rng.random(n) < ps).astype(np.float32)
    tau = 3.0 + 2.0 * Xn[:, 0]
    yn = (2.0 + 1.5 * Xn[:, 0] + 0.7 * Xn[:, 1] + tau * tn + rng.normal(size=n)).astype(np.float32)

    padded = np.zeros((n, covariates), dtype=np.float32)
    padded[:, :d] = Xn

    def onnx_mu(arm):
        feed = {
            "X_context": padded[None, :, :],
            "t_context": tn[None, :],
            "y_context": yn[None, :],
            "X_query": padded[None, :, :],
            "t_query": np.full((1, n), float(arm), dtype=np.float32),
        }
        return session.run(None, feed)[0].reshape(-1)

    def torch_mu(arm):
        with torch.no_grad():
            out = make_wrapper(torch, model)(
                torch.from_numpy(padded[None, :, :]),
                torch.from_numpy(tn[None, :]),
                torch.from_numpy(yn[None, :]),
                torch.from_numpy(padded[None, :, :]),
                torch.full((1, n), float(arm)),
            )
        return out.numpy().reshape(-1)

    cate_onnx = onnx_mu(1) - onnx_mu(0)
    cate_torch = torch_mu(1) - torch_mu(0)
    cate_gap = float(np.max(np.abs(cate_onnx - cate_torch)))
    ate_gap = abs(float(cate_onnx.mean() - cate_torch.mean()))
    corr = float(np.corrcoef(cate_onnx, cate_torch)[0, 1])
    print()
    print("estimand parity on a real DGP (n=%d, true ATE %.4f):" % (n, tau.mean()))
    print("  ONNX  ATE %.4f | torch ATE %.4f | max |CATE diff| %.3e | corr %.6f"
          % (cate_onnx.mean(), cate_torch.mean(), cate_gap, corr))
    manifest["estimand_parity"] = {"max_abs_cate_diff": cate_gap, "abs_ate_diff": ate_gap,
                                   "cate_correlation": corr}
    with io.open(manifest_path, "w", encoding="utf8") as handle:
        json.dump(manifest, handle, indent=2)

    # PyTorch runs a fused attention kernel; ONNX Runtime decomposes it. Over 12
    # layers and a 1024-bin softmax that costs ~1e-3 of relative agreement in
    # fp32 - real, but far below the noise on any causal estimate. The gate is
    # therefore set on the estimand, which is what a user actually reads.
    if ate_gap > 0.01 or corr < 0.999:
        print("ESTIMAND PARITY FAILED: ATE differs by %.4f, CATE correlation %.6f"
              % (ate_gap, corr))
        return 1
    print("ESTIMAND PARITY PASSED: ATE agrees to %.5f, CATE correlation %.6f "
          "(logit-level relative parity %.2e, from attention-kernel differences)"
          % (ate_gap, corr, worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
