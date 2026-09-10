"""Export Do-PFN to ONNX for the DuckDo inference runtime (roadmap phase 5).

Do-PFN is an in-context learner: it takes a labelled context and a set of query
rows and returns logits over a 100-bucket bar distribution of the interventional
outcome. Column 0 of X is the treatment. CATE is the difference between the
distribution means under do(T=1) and do(T=0).

## Why the context length is fixed and the query length is not

The model computes its context/query split as `single_eval_pos = len(context)`,
a Python int. The torchscript tracer bakes that in as a constant, so a graph
exported at one context length silently returns the wrong rows at another - not
a shape error, just wrong numbers, which is the worst kind of bug to ship.

The dynamo exporter would carry it symbolically, but it cannot get past the
model's data-dependent ModuleList indexing without upstream surgery we are not
in a position to verify.

So we export a *ladder* of graphs at fixed context sizes, leaving the query
length dynamic. With the context fixed, the baked-in split point is exactly
right, and the query axis traces cleanly. This is not a workaround forced on the
design: Do-PFN's context caps at 2200 rows, so DuckDo has to subsample context
for any real table regardless. The runtime picks the largest rung that fits.

Weights are written as external data, so the extension can compile in a
weight-free graph and download the weights separately under the model's own
licence - which is none: the upstream repository states no licence, so the
weights are all rights reserved and are exported for your own use only.

Dev-only; the extension ships the graph, never the weights.

    python scripts/export/export_dopfn.py --repo <path-to-Do-PFN-clone> --out build/models
"""

import argparse
import io
import json
import os
import sys

import numpy as np

DEFAULT_LADDER = (128, 512, 1024, 2048)
# All rungs are the same model, so they share one initializer blob.
SHARED_WEIGHTS = "dopfn.weights.bin"
TOLERANCE = 1e-4


def shim_torch_for_old_pickles(torch):
    """The checkpoint was pickled under torch 2.1, whose modules re-exported a
    handful of typing names that later versions dropped. Unpickling resolves them
    by module path, so put them back before loading."""
    import typing
    import torch.nn.modules.activation as act
    import torch.nn.modules.linear as lin
    import torch.nn.modules.transformer as tf

    names = {
        "Optional": typing.Optional, "Union": typing.Union, "Any": typing.Any,
        "Callable": typing.Callable, "List": typing.List, "Tuple": typing.Tuple,
        "Dict": typing.Dict, "Tensor": torch.Tensor,
    }
    for module in (tf, act, lin):
        for name, value in names.items():
            if not hasattr(module, name):
                setattr(module, name, value)


def patch_nansum(torch):
    """ONNX opset 17 has no aten::nansum. It is exactly sum(where(isnan, 0, x)),
    which is expressible, so swap in that definition before tracing. A faithful
    rewrite, not an approximation - the parity check confirms it."""

    def nansum(input, axis=None, dim=None, keepdim=False, **kwargs):
        cleaned = torch.where(torch.isnan(input), torch.zeros_like(input), input)
        reduce_dim = dim if dim is not None else axis
        if reduce_dim is None:
            return torch.sum(cleaned, **kwargs)
        return torch.sum(cleaned, dim=reduce_dim, keepdim=keepdim, **kwargs)

    torch.nansum = nansum


def load(repo):
    import torch

    shim_torch_for_old_pickles(torch)
    patch_nansum(torch)
    os.chdir(repo)
    sys.path.insert(0, repo)
    from scripts.transformer_prediction_interface.model_builder import load_model

    model, config = load_model(path=None, device="cpu", verbose=False)
    model.eval()
    return torch, model, config


def make_wrapper(torch, model):
    """Expose a plain three-tensor forward for the tracer."""

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, context_x, context_y, query_x):
            # (context, batch, features), (context, batch), (query, batch, features)
            return self.inner(context_x, context_y, query_x, only_return_standard_out=True)

    return Wrapper()


def as_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def export_one(torch, wrapper, features, context_len, out_dir, opset):
    """Export one rung and prove the query axis really is dynamic."""
    import onnx
    import onnxruntime as ort

    torch.manual_seed(0)
    context_x = torch.randn(context_len, 1, features)
    context_y = torch.randn(context_len, 1)
    query_x = torch.randn(16, 1, features)

    with torch.no_grad():
        reference = as_tensor(wrapper(context_x, context_y, query_x))

    stem = "dopfn_ctx%d" % context_len
    path = os.path.join(out_dir, stem + ".onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (context_x, context_y, query_x),
            path,
            input_names=["context_x", "context_y", "query_x"],
            output_names=["logits"],
            dynamic_axes={"query_x": {0: "query"}, "logits": {0: "query"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # Split weights out so the extension can ship a weight-free graph. Every
    # rung is the same model, so they share one blob - but save_model APPENDS to
    # an existing external-data file, so it has to be removed first.
    shared = os.path.join(out_dir, SHARED_WEIGHTS)
    if os.path.exists(shared):
        os.remove(shared)
    proto = onnx.load(path)
    onnx.save_model(
        proto, path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=SHARED_WEIGHTS,
        size_threshold=1024,
        convert_attribute=False,
    )

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(None, {
        "context_x": context_x.numpy(),
        "context_y": context_y.numpy(),
        "query_x": query_x.numpy(),
    })[0]
    gap = float(np.max(np.abs(got - reference.numpy())))

    # A different query length, same context: the axis that must stay live.
    query2 = torch.randn(41, 1, features)
    with torch.no_grad():
        ref2 = as_tensor(wrapper(context_x, context_y, query2))
    got2 = session.run(None, {
        "context_x": context_x.numpy(),
        "context_y": context_y.numpy(),
        "query_x": query2.numpy(),
    })[0]
    if got2.shape != tuple(ref2.shape):
        raise RuntimeError("query axis is not dynamic at context=%d: onnx %s vs torch %s"
                           % (context_len, got2.shape, tuple(ref2.shape)))
    gap2 = float(np.max(np.abs(got2 - ref2.numpy())))

    graph_bytes = os.path.getsize(path)
    weight_bytes = os.path.getsize(os.path.join(out_dir, SHARED_WEIGHTS))
    print("  context=%-5d graph=%7.1f KB  weights=%6.2f MB  parity=%.2e  parity@query41=%.2e"
          % (context_len, graph_bytes / 1024.0, weight_bytes / 1e6, gap, gap2))
    return {
        "context": context_len,
        "graph": os.path.basename(path),
        "weights": SHARED_WEIGHTS,
        "graph_bytes": graph_bytes,
        "weight_bytes": weight_bytes,
        "parity_max_abs_diff": gap,
        "parity_max_abs_diff_query41": gap2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="path to a Do-PFN checkout")
    parser.add_argument("--out", default=os.path.join("build", "models"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--ladder", default=",".join(str(c) for c in DEFAULT_LADDER))
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    repo = os.path.abspath(args.repo)
    os.makedirs(out_dir, exist_ok=True)
    ladder = [int(c) for c in args.ladder.split(",") if c.strip()]

    torch, model, config = load(repo)
    features = int(config["max_num_features"])
    buckets = int(config["num_buckets"])
    max_context = int(config["seq_len"])
    print("loaded Do-PFN: %.2fM params, max_num_features=%d, num_buckets=%d, seq_len=%d"
          % (sum(p.numel() for p in model.parameters()) / 1e6, features, buckets, max_context))

    wrapper = make_wrapper(torch, model)
    print("exporting ladder %s to %s" % (ladder, out_dir))
    rungs = []
    for context_len in ladder:
        if context_len >= max_context:
            print("  context=%d skipped: at or above the model's seq_len" % context_len)
            continue
        rungs.append(export_one(torch, wrapper, features, context_len, out_dir, args.opset))

    borders = None
    criterion = getattr(model, "criterion", None)
    if criterion is not None and hasattr(criterion, "borders"):
        borders = criterion.borders.detach().cpu().numpy().astype(np.float64).tolist()

    manifest = {
        "model": "dopfn",
        "source": "https://github.com/jr2021/Do-PFN",
        "license": "none stated upstream",
        "attribution_required": False,
        "setting": "non-identifiable prior",
        "opset": args.opset,
        "max_features": features,
        "num_buckets": buckets,
        "model_max_context": max_context,
        "treatment_column": 0,
        "inputs": ["context_x", "context_y", "query_x"],
        "output": "logits",
        "dynamic_axes": {"query_x": ["query"], "logits": ["query"]},
        "weights": SHARED_WEIGHTS,
        "weight_bytes": rungs[0]["weight_bytes"] if rungs else 0,
        "context_ladder": rungs,
        "bar_distribution_borders": borders,
        "notes": (
            "Context length is fixed per rung because the model's context/query "
            "split point is a Python int the tracer bakes in. The query length is "
            "dynamic. Pick the largest rung whose context is <= the number of "
            "available context rows."
        ),
    }
    manifest_path = os.path.join(out_dir, "dopfn.manifest.json")
    with io.open(manifest_path, "w", encoding="utf8") as handle:
        json.dump(manifest, handle, indent=2)
    print("wrote", manifest_path)

    if not rungs:
        print("NO RUNGS EXPORTED")
        return 1

    # The rungs were written one at a time, each replacing the shared weight
    # blob. Re-verify every graph against the file that actually survived.
    print("re-verifying every rung against the surviving weight blob:")
    import onnxruntime as ort
    for rung in rungs:
        context_len = rung["context"]
        torch.manual_seed(0)
        cx = torch.randn(context_len, 1, features)
        cy = torch.randn(context_len, 1)
        qx = torch.randn(23, 1, features)
        with torch.no_grad():
            ref = as_tensor(wrapper(cx, cy, qx))
        session = ort.InferenceSession(os.path.join(out_dir, rung["graph"]),
                                       providers=["CPUExecutionProvider"])
        got = session.run(None, {"context_x": cx.numpy(), "context_y": cy.numpy(),
                                 "query_x": qx.numpy()})[0]
        gap = float(np.max(np.abs(got - ref.numpy())))
        rung["parity_final"] = gap
        status = "ok" if gap <= TOLERANCE else "FAILED"
        print("  context=%-5d parity=%.2e  %s" % (context_len, gap, status))
    worst = max(max(r["parity_max_abs_diff"], r["parity_max_abs_diff_query41"],
                    r.get("parity_final", 0.0)) for r in rungs)
    if worst > TOLERANCE:
        print("PARITY FAILED: worst %.3e exceeds %.0e" % (worst, TOLERANCE))
        return 1
    print("PARITY PASSED: worst %.3e over %d rungs, tolerance %.0e"
          % (worst, len(rungs), TOLERANCE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
