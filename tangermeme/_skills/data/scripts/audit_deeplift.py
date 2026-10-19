# audit_deeplift.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Check whether DeepLIFT/SHAP attributions of a model can be trusted.

Two checks, both needed. ``audit`` walks one forward pass and lists every
torch operation that no DeepLIFT rule covers, attributed to the innermost
module running it: a functional ``softmax``, a ``*`` between two activations,
a custom module with no rule. ``check_convergence`` runs ``deep_lift_shap``
and measures summation-to-delta itself, relative to the scale of the output
differences, because the library's warning compares each delta against an
absolute threshold and a model with small outputs can be badly wrong in
silence.

Importable::

	from audit_deeplift import audit, check_convergence
	audit(model, X)                       # -> [(module type, op), ...]
	check_convergence(model, X, target=0) # -> dict

Or from the shell, with the model built by a Python expression evaluated after
importing a module::

	python audit_deeplift.py --model my_models:GatedConv() --length 200 --target 1
	python audit_deeplift.py --model my_models:load("ckpt.pt") --length 2114

Exit status is 1 when the relative convergence error exceeds ``--tol``.
"""

import argparse
import importlib
import sys
import warnings

import torch

from tangermeme.deep_lift_shap import BilinearOp
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.utils import random_one_hot


RULED = {
	torch.nn.ReLU, torch.nn.ReLU6, torch.nn.RReLU, torch.nn.SELU,
	torch.nn.CELU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Mish, torch.nn.GLU,
	torch.nn.ELU, torch.nn.LeakyReLU, torch.nn.Sigmoid, torch.nn.Tanh,
	torch.nn.Softplus, torch.nn.Softshrink, torch.nn.LogSigmoid,
	torch.nn.PReLU, torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.Softmax,
	torch.nn.LayerNorm, torch.nn.RMSNorm, BilinearOp,
}

LINEAR = {
	"linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "add", "sub",
	"cat", "stack", "reshape", "view", "permute", "transpose", "flatten",
	"unsqueeze", "squeeze", "chunk", "split", "sum", "mean", "pad", "to",
	"type", "clone", "detach", "contiguous", "dim", "size", "dropout",
	"avg_pool1d", "avg_pool2d", "embedding", "batch_norm", "expand",
	"repeat", "narrow", "index_select", "roll", "__get__", "__getitem__",
	"_set_grad_enabled", "empty", "zeros", "ones", "arange", "t", "float",
	"half", "double", "cpu", "cuda", "numel", "shape", "is_floating_point",
	"masked_fill", "__setitem__", "neg", "flip",
}


def audit(model, X, extra_rules=()):
	"""Return every (module type, op) in the forward pass with no DeepLIFT rule.

	``mul``, ``div``, ``matmul``, ``bmm`` and ``einsum`` are reported on purpose:
	a product with a constant is linear, a product of two activations is not,
	and only the source tells them apart.
	"""

	ruled = RULED | set(extra_rules)
	stack, found = [], set()

	def _push(module, inputs):
		stack.append(module)

	def _pop(module, inputs, outputs):
		stack.pop()

	class _Trace(torch.overrides.TorchFunctionMode):
		def __torch_function__(self, func, types, args=(), kwargs=None):
			name = getattr(func, "__name__", "?")
			if name not in LINEAR and stack and type(stack[-1]) not in ruled:
				found.add((type(stack[-1]).__name__, name))
			return func(*args, **(kwargs or {}))

	hooks = [
		torch.nn.modules.module.register_module_forward_pre_hook(_push),
		torch.nn.modules.module.register_module_forward_hook(_pop),
	]
	try:
		with _Trace(), torch.no_grad():
			model(X)
	finally:
		for h in hooks:
			h.remove()

	return sorted(found)


def check_convergence(model, X, target=0, n_shuffles=10, random_state=0,
	device="cpu", **kwargs):
	"""Measure summation-to-delta relative to the output scale.

	Returns a dict with the mean absolute delta, the scale of the output
	differences it is compared against, their ratio, and the number of
	convergence warnings the library raised.
	"""

	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		attr, refs = deep_lift_shap(model, X, target=target,
			n_shuffles=n_shuffles, random_state=random_state,
			return_references=True, device=device, **kwargs)

	n_warn = sum(issubclass(c.category, RuntimeWarning) for c in caught)
	model = model.to(device).eval()

	with torch.no_grad():
		y = model(X.to(device).float())[:, target].float()
		y_ref = torch.stack([model(refs[:, k].to(device).float())[:, target]
			for k in range(refs.shape[1])], dim=1).float().mean(dim=1)

	delta = attr.sum(dim=(1, 2)).to(device) - (y - y_ref)
	scale = (y - y_ref).abs().mean().item()
	mean_delta = delta.abs().mean().item()
	return {
		"mean_abs_delta": mean_delta,
		"output_difference_scale": scale,
		"relative_error": mean_delta / max(scale, 1e-12),
		"n_warnings": n_warn,
	}


def _build(spec):
	module_name, expr = spec.split(":", 1)
	module = importlib.import_module(module_name)
	return eval(expr, vars(module))


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	parser.add_argument("--model", required=True,
		help="module:expression building the model, e.g. my_models:Net()")
	parser.add_argument("--length", type=int, default=200)
	parser.add_argument("--alphabet-size", type=int, default=4)
	parser.add_argument("--n", type=int, default=8)
	parser.add_argument("--target", type=int, default=0)
	parser.add_argument("--tol", type=float, default=1e-3,
		help="largest acceptable relative convergence error")
	parser.add_argument("--device", default="cpu")
	args = parser.parse_args(argv)

	sys.path.insert(0, ".")
	model = _build(args.model).eval()
	X = random_one_hot((args.n, args.alphabet_size, args.length),
		random_state=0).float()

	print("unhooked ops (module type, op):")
	ops = audit(model, X)
	for owner, op in ops:
		print("  {}: {}".format(owner, op))
	if not ops:
		print("  none")

	r = check_convergence(model, X, target=args.target, device=args.device)
	print("convergence: mean |delta| {:.3e} against output differences of "
		"{:.3e}, relative error {:.3e}, {} library warnings".format(
		r["mean_abs_delta"], r["output_difference_scale"],
		r["relative_error"], r["n_warnings"]))

	if r["relative_error"] > args.tol:
		print("FAIL: attributions do not sum to the output difference")
		return 1
	print("OK")
	return 0


if __name__ == "__main__":
	sys.exit(main())
