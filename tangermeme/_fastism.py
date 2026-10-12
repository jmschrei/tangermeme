# _fastism.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Private machinery for `saturation_mutagenesis(fast=True)`.

fastISM (Nair et al., Bioinformatics 2022) speeds up in silico saturation
mutagenesis by noting that, in a convolutional network, a single-base edit
changes only a bounded span of each intermediate layer. This module applies
that idea to any PyTorch model that `torch.export` can capture. The captured
graph of ATen operations is split into four kinds of node:

- static: depends on neither the sequence nor the batch, such as parameters.
  Computed once.
- batched: depends on the extra `args` or on the batch size, but not on the
  sequence. Computed once per example and batch size.
- local: depends on the sequence, and each output position along the length
  axis depends on a bounded range of input positions. For each edit, only the
  window of positions the edit can change is computed; every other position
  is read from the activations cached for the unedited sequence.
- global: everything else that depends on the sequence. Computed at full
  length for each edit, from the local windows spliced into the cached
  activations.

An operation is local only when a rule below covers it, so an operation
without a rule is computed at full length rather than approximated.
"""

from __future__ import annotations

import contextlib
import logging
import operator
import warnings

import numpy
import torch

from torch.fx.node import map_arg
from torch.fx.node import Node
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_unflatten


_STATIC = 0
_BATCHED = 1
_LOCAL = 2
_GLOBAL = 3

_INF = 1 << 40

# Loggers that print a traceback when torch.export fails. The failure is
# reported to the user as the reason fast=True was not used.
_EXPORT_LOGGERS = ("torch._subclasses.fake_tensor",
	"torch.fx.experimental.symbolic_shapes", "torch._export",
	"torch.export", "torch._dynamo")


class _Unsupported(Exception):
	"""Raised when a model cannot be run on the fast path, with the reason."""


###


_ELEMENTWISE = {
	"aten.abs.default", "aten.acos.default", "aten.asin.default",
	"aten.atan.default", "aten.ceil.default", "aten.cos.default",
	"aten.cosh.default", "aten.erf.default", "aten.erfc.default",
	"aten.exp.default", "aten.exp2.default", "aten.expm1.default",
	"aten.floor.default", "aten.frac.default", "aten.log.default",
	"aten.log10.default", "aten.log1p.default", "aten.log2.default",
	"aten.logical_not.default", "aten.neg.default", "aten.reciprocal.default",
	"aten.relu.default", "aten.round.default", "aten.rsqrt.default",
	"aten.sigmoid.default", "aten.sign.default", "aten.sgn.default",
	"aten.sin.default", "aten.sinh.default", "aten.sqrt.default",
	"aten.square.default", "aten.tan.default", "aten.tanh.default",
	"aten.trunc.default", "aten.gelu.default", "aten.silu.default",
	"aten.mish.default", "aten.elu.default", "aten.celu.default",
	"aten.selu.default", "aten.leaky_relu.default", "aten.hardtanh.default",
	"aten.hardswish.default", "aten.hardsigmoid.default",
	"aten.softplus.default", "aten.threshold.default", "aten.logit.default",
	"aten.nan_to_num.default", "aten.clamp.default", "aten.clamp.Tensor",
	"aten.clamp_min.default", "aten.clamp_max.default",
	"aten.bitwise_not.default", "aten.isnan.default", "aten.isinf.default",
	"aten._to_copy.default", "aten.to.dtype", "aten.to.dtype_layout",
	"aten.clone.default", "aten.alias.default", "aten.detach.default",
	"aten.lift_fresh_copy.default", "aten.zeros_like.default",
	"aten.ones_like.default", "aten.full_like.default",
	"aten.add.Tensor", "aten.add.Scalar", "aten.sub.Tensor",
	"aten.sub.Scalar", "aten.rsub.Tensor", "aten.rsub.Scalar",
	"aten.mul.Tensor", "aten.mul.Scalar", "aten.div.Tensor",
	"aten.div.Scalar", "aten.div.Tensor_mode", "aten.div.Scalar_mode",
	"aten.pow.Tensor_Tensor", "aten.pow.Tensor_Scalar", "aten.pow.Scalar",
	"aten.maximum.default", "aten.minimum.default", "aten.fmax.default",
	"aten.fmin.default", "aten.atan2.default", "aten.remainder.Tensor",
	"aten.remainder.Scalar", "aten.fmod.Tensor", "aten.fmod.Scalar",
	"aten.copysign.Tensor", "aten.hypot.default", "aten.xlogy.Tensor",
	"aten.eq.Tensor", "aten.eq.Scalar", "aten.ne.Tensor", "aten.ne.Scalar",
	"aten.gt.Tensor", "aten.gt.Scalar", "aten.ge.Tensor", "aten.ge.Scalar",
	"aten.lt.Tensor", "aten.lt.Scalar", "aten.le.Tensor", "aten.le.Scalar",
	"aten.logical_and.default", "aten.logical_or.default",
	"aten.logical_xor.default", "aten.bitwise_and.Tensor",
	"aten.bitwise_or.Tensor", "aten.bitwise_xor.Tensor",
	"aten.where.self", "aten.where.ScalarSelf", "aten.where.ScalarOther",
	"aten.masked_fill.Scalar", "aten.masked_fill.Tensor",
	"aten.lerp.Scalar", "aten.lerp.Tensor", "aten.addcmul.default",
	"aten.addcdiv.default",
}

_REDUCTIONS = {
	"aten.sum.dim_IntList", "aten.mean.dim", "aten.amax.default",
	"aten.amin.default", "aten.logsumexp.default", "aten.var.correction",
	"aten.std.correction", "aten.prod.dim_int", "aten.any.dim",
	"aten.any.dims", "aten.all.dim", "aten.all.dims", "aten.argmax.default",
	"aten.argmin.default", "aten.max.dim", "aten.min.dim",
}

_ALONG_DIM = {
	"aten.softmax.int": "dim", "aten._softmax.default": "dim",
	"aten.log_softmax.int": "dim", "aten._log_softmax.default": "dim",
	"aten.cumsum.default": "dim", "aten.cumprod.default": "dim",
	"aten.glu.default": "dim", "aten.index_select.default": "dim",
}

_RESHAPES = {
	"aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
	"aten.flatten.using_ints", "aten.unflatten.int",
}

_SPLITS = {
	"aten.split.Tensor", "aten.split_with_sizes.default", "aten.chunk.default",
	"aten.unbind.int",
}

_CONVS = {
	"aten.conv1d.default", "aten.conv1d.padding", "aten.conv2d.default",
	"aten.conv2d.padding", "aten.conv3d.default", "aten.conv3d.padding",
	"aten.convolution.default",
}

_POOLS = {
	"aten.max_pool1d.default", "aten.max_pool2d.default",
	"aten.avg_pool1d.default", "aten.avg_pool2d.default",
}

_SHAPE_QUERIES = {"aten.sym_size.int", "aten.sym_numel.default"}


###


def _meta(node):
	"""Return the example value `torch.export` recorded for a node."""

	return node.meta.get("val") if isinstance(node, Node) else None


def _bind(node):
	"""Map each argument of an ATen call to its name, filling in defaults."""

	bound = {}
	for i, arg in enumerate(node.target._schema.arguments):
		if i < len(node.args):
			bound[arg.name] = node.args[i]
		elif arg.name in node.kwargs:
			bound[arg.name] = node.kwargs[arg.name]
		elif arg.has_default_value():
			bound[arg.name] = arg.default_value
		else:
			bound[arg.name] = None
	return bound


def _leaves(node):
	"""Return every node an operation reads, in argument order."""

	leaves = []
	map_arg((node.args, node.kwargs), lambda n: leaves.append(n))
	return leaves


def _norm(dim, ndim):
	return dim + ndim if dim < 0 else dim


def _per_dim(value, n):
	"""Expand an int or a one-element list to a list of length n."""

	if isinstance(value, int):
		return [value] * n
	value = list(value)
	return value * n if len(value) == 1 else value


def _batched_shape(value):
	"""Return the shape of a tensor whose only symbolic dimension is a leading
	batch dimension, with that dimension as None, or None otherwise."""

	if not isinstance(value, torch.Tensor) or value.dim() == 0:
		return None
	if not isinstance(value.shape[0], torch.SymInt):
		return None

	shape = [None]
	for d in value.shape[1:]:
		if not isinstance(d, int):
			return None
		shape.append(d)
	return shape


def _concrete(size, b):
	"""Evaluate a size recorded by `torch.export` at batch size b.

	The batch dimension is the only dynamic one, so every symbol in a size is
	a batch size.
	"""

	if isinstance(size, int):
		return size

	expr = size.node.expr
	return int(expr.subs({symbol: b for symbol in expr.free_symbols}))


def _has_hooks(model):
	"""Return True when forward hooks could run code the graph does not hold."""

	module = torch.nn.modules.module
	if module._global_forward_hooks or module._global_forward_pre_hooks:
		return True

	for m in model.modules():
		if m._forward_hooks or m._forward_pre_hooks:
			return True
	return False


###


class _Aligned:
	"""A rule for an operation where output position j along the length axis
	depends only on position j of each input along its length axis.

	`data` maps each input that has to be read over the output window to
	(axis, ndim): the input is first right-aligned to `ndim` dimensions, as
	broadcasting would, and then read along `axis`. Inputs not in `data` are
	passed to the operation unchanged.
	"""

	def __init__(self, node, lax, length, data, call=None, element_lax=None):
		self.node = node
		self.lax = lax
		self.length = length
		self.data = data
		self._call = call
		self.element_lax = element_lax

	def interval(self, fi):
		return _hull([(fi.lo[n], fi.hi[n]) for n in self.data
			if fi.kind[n] == _LOCAL])

	def ranges(self, starts, width):
		return {n: (starts, width, 0) for n in self.data}

	def call(self, get, b, width):
		if self._call is not None:
			return self._call(get, b, width)

		node = self.node
		return node.target(*map_arg(node.args, get),
			**map_arg(node.kwargs, get))


class _Window:
	"""A rule for an operation where output position j along the length axis
	reads input positions j*stride - offset + i*dilation for i in
	range(kernel): convolution, pooling, constant padding, and slicing.

	The input is read over the positions an output window needs, with `fill`
	standing in for positions outside the input, so `call` runs the operation
	without padding along the length axis.
	"""

	def __init__(self, node, source, lax, length, stride, dilation, kernel,
		offset, fill, call):
		self.node = node
		self.source = source
		self.lax = lax
		self.length = length
		self.stride = stride
		self.dilation = dilation
		self.kernel = kernel
		self.offset = offset
		self.fill = fill
		self._call = call
		self.data = {source: None}
		self.element_lax = None

	def interval(self, fi):
		lo, hi = fi.lo[self.source], fi.hi[self.source]
		s, d, k, p = self.stride, self.dilation, self.kernel, self.offset

		out_lo = -((-(lo + p - (k - 1) * d)) // s)
		out_hi = (hi - 1 + p) // s + 1
		out_lo = numpy.clip(out_lo, 0, self.length)
		out_hi = numpy.clip(out_hi, 0, self.length)

		empty = (lo >= hi) | (out_lo >= out_hi)
		return (numpy.where(empty, _INF, out_lo),
			numpy.where(empty, -_INF, out_hi))

	def ranges(self, starts, width):
		s, d, k, p = self.stride, self.dilation, self.kernel, self.offset
		return {self.source: (starts * s - p, (width - 1) * s + (k - 1) * d + 1,
			self.fill)}

	def call(self, get, b, width):
		return self._call(get, width)


class _Flip:
	"""A rule for `flip` along the length axis, which maps output position j
	to input position length - 1 - j."""

	def __init__(self, node, source, lax, length, dims):
		self.node = node
		self.source = source
		self.lax = lax
		self.length = length
		self.dims = dims
		self.data = {source: None}
		self.element_lax = None

	def interval(self, fi):
		lo, hi = fi.lo[self.source], fi.hi[self.source]
		empty = lo >= hi
		return (numpy.where(empty, _INF, self.length - hi),
			numpy.where(empty, -_INF, self.length - lo))

	def ranges(self, starts, width):
		return {self.source: (self.length - starts - width, width, 0)}

	def call(self, get, b, width):
		return torch.ops.aten.flip.default(get(self.source), self.dims)


class _GetItem:
	"""A rule for selecting one output of a local operation that returns
	several tensors along the same length axis."""

	def __init__(self, node, source, index, lax, length):
		self.node = node
		self.source = source
		self.index = index
		self.lax = lax
		self.length = length
		self.data = {}
		self.element_lax = None

	def interval(self, fi):
		return fi.lo[self.source], fi.hi[self.source]


def _hull(intervals):
	"""Return the smallest interval holding every interval, per position."""

	lo = numpy.minimum.reduce([i[0] for i in intervals])
	hi = numpy.maximum.reduce([i[1] for i in intervals])
	return lo, hi


def _right_align(x, ndim):
	"""Prepend size-one dimensions so that x has ndim dimensions, as
	broadcasting would."""

	if x.dim() < ndim:
		return x.reshape((1,) * (ndim - x.dim()) + tuple(x.shape))
	return x


class _Read:
	"""How a local operation reads one input over its output window.

	For each edit position, the input is read over `width` positions along
	`axis` starting at `starts`, after right-aligning it to `ndim`
	dimensions. Positions outside the input read as `fill`. In "direct" mode
	the read is exactly the input's own window. In "mixed" mode the input's
	window starts `offsets` positions into the read and the rest comes from
	the cached reference. In "const" mode the input does not depend on X and
	everything comes from its value.
	"""

	def __init__(self, source, axis, ndim, starts, width, fill, length, mode,
		window_starts, device):
		self.source = source
		self.axis = axis
		self.ndim = ndim
		self.width = width
		self.fill = fill
		self.mode = mode

		# Pad the reference once so that every read is in range.
		self.left = int(max(0, -starts.min()))
		self.right = int(max(0, (starts + width).max() - length))
		self.starts = torch.from_numpy(starts + self.left).to(device)

		if mode == "mixed":
			self.offsets_np = window_starts - starts
			self.offsets = torch.from_numpy(self.offsets_np).to(device)


###


class _FastISM:
	"""Run a model on single-base edits of a sequence, computing only what
	each edit can change.

	The model is captured once with `torch.export` and classified into
	static, batched, local and global operations (see the module docstring).
	`predict_edits` then returns the model's output for a list of edits of
	one example, matching `predict` on the edited sequences up to floating
	point rounding.

	The caller is responsible for moving the model to `device`, putting it in
	evaluation mode, and disabling gradients, as `predict` does.


	Parameters
	----------
	model: torch.nn.Module
		The model, already on `device` and in evaluation mode.

	X: torch.Tensor, shape=(-1, len(alphabet), length)
		The one-hot encoded sequences, as int8 on the CPU.

	args: tuple or None
		Additional per-example inputs to the model, batched along their first
		dimension like `X`, or None.

	start: int
		The first position that will be edited.

	end: int
		The position after the last one that will be edited.

	device: torch.device
		The device to run the model on.

	dtype: torch.dtype
		The dtype that `X` and `args` are cast to, as in `predict`.

	use_autocast: bool
		Whether to run the model under `torch.autocast` with `dtype`.
	"""

	def __init__(self, model, X, args, start, end, device, dtype,
		use_autocast):
		if not hasattr(torch.export, "Dim") or not hasattr(torch.export.Dim,
			"AUTO"):
			raise _Unsupported("torch {} does not provide "
				"torch.export.Dim.AUTO".format(torch.__version__))

		model = getattr(model, "_orig_mod", model)
		if _has_hooks(model):
			raise _Unsupported("the model has forward hooks registered, which "
				"the captured graph would not run")

		self.X = X
		self.args = args
		self.start = start
		self.end = end
		self.device = device
		self.dtype = dtype
		self.use_autocast = use_autocast
		self.alphabet = X.shape[1]

		self._export(model)
		self._classify()
		self._regions()
		self._static_values()

		self._example = None
		self._aranges = {}

	def _autocast(self):
		if self.use_autocast:
			return torch.autocast(device_type=self.device.type,
				dtype=self.dtype)
		return contextlib.nullcontext()

	def _cast(self, X):
		return X.type(self.dtype).to(self.device)

	def _export(self, model):
		"""Capture the model as a functional graph of ATen operations with a
		dynamic batch dimension.

		The graph is captured with the inputs in the dtype they are given to the
		model, and under autocast when the model runs under it, so that the
		casts autocast and the model make are recorded in it. A graph captured
		in another dtype would hold the dtype checks of that dtype instead, and
		decomposing it outside autocast would trace operations on mismatched
		dtypes.
		"""

		inputs = [self.X[:1].repeat(2, 1, 1)]
		if self.args is not None:
			inputs += [a[:1].repeat(2, *(1 for _ in a.shape[1:]))
				for a in self.args]
		inputs = tuple(self._cast(x) for x in inputs)
		dynamic = tuple({0: torch.export.Dim.AUTO} for _ in inputs)

		loggers = [logging.getLogger(name) for name in _EXPORT_LOGGERS]
		levels = [logger.level for logger in loggers]
		try:
			for logger in loggers:
				logger.setLevel(logging.CRITICAL)

			with warnings.catch_warnings():
				# Raised from within torch.export's own copy of its graph.
				warnings.filterwarnings("ignore", category=FutureWarning,
					message=r".*isinstance\(treespec, LeafSpec\)")

				with self._autocast():
					ep = torch.export.export(model, inputs,
						dynamic_shapes=dynamic, strict=False)
					ep = ep.run_decompositions(decomp_table={})
		except Exception as e:
			message = str(e).strip().split("\n")[0]
			raise _Unsupported("torch.export could not capture the model "
				"({}: {})".format(type(e).__name__, message)) from e
		finally:
			for logger, level in zip(loggers, levels):
				logger.setLevel(level)

		for spec in ep.graph_signature.output_specs:
			if spec.kind.name != "USER_OUTPUT":
				raise _Unsupported("the model's forward pass modifies its own "
					"state ({} {})".format(spec.kind.name, spec.target))

		self.ep = ep
		self.graph = ep.graph
		self.out_spec = ep.call_spec.out_spec

		self.placeholders = {}
		user = []
		for spec in ep.graph_signature.input_specs:
			kind = spec.kind.name
			if kind == "USER_INPUT":
				user.append(spec.arg.name)
			elif kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR",
				"CUSTOM_OBJ"):
				if spec.target in ep.state_dict:
					value = ep.state_dict[spec.target]
				else:
					value = ep.constants[spec.target]
				self.placeholders[spec.arg.name] = value
			else:
				raise _Unsupported("the model's captured graph takes a {} "
					"input".format(kind))

		nodes = {n.name: n for n in self.graph.nodes}
		self.x = nodes[user[0]]
		self.arg_nodes = [nodes[name] for name in user[1:]]
		self.output = [n for n in self.graph.nodes if n.op == "output"][0]

		if _batched_shape(_meta(self.x)) is None:
			raise _Unsupported("the model's forward pass depends on the batch "
				"size")

	###

	def _classify(self):
		"""Assign every node a kind, and every local node a rule."""

		self.kind = {}
		self.lax = {}
		self.length = {}
		self.rule = {}
		self.local_order = []
		self.global_order = []
		self.batched_order = []
		self.static_order = []

		for node in self.graph.nodes:
			if node.op == "output":
				continue

			if node is self.x:
				kind = _LOCAL
				self.lax[node] = 2
				self.length[node] = _meta(node).shape[2]
			elif node.op == "placeholder":
				kind = _BATCHED if node in self.arg_nodes else _STATIC
			elif node.op == "get_attr":
				kind = _STATIC
			else:
				kinds = [self.kind[n] for n in _leaves(node)]
				if str(node.target) in _SHAPE_QUERIES:
					# The size of an edited tensor depends on the batch size
					# but not on the sequence.
					kinds = [_BATCHED if k >= _LOCAL else k for k in kinds]

				kind = max(kinds, default=_STATIC)
				if kind >= _LOCAL:
					kind = self._classify_varying(node, kinds)

			self.kind[node] = kind
			{_STATIC: self.static_order, _BATCHED: self.batched_order,
				_LOCAL: self.local_order, _GLOBAL: self.global_order}[
				kind].append(node)

		outputs = [n for n in tree_flatten(self.output.args[0])[0]
			if isinstance(n, Node)]
		self.frontier = [n for n in self.local_order
			if n in outputs or any(self.kind[u] == _GLOBAL for u in n.users)]

		self.local_ops = sorted({str(n.target) for n in self.local_order
			if n.op == "call_function"})

	def _classify_varying(self, node, kinds):
		"""Return _LOCAL if a rule covers this operation, else _GLOBAL."""

		if _GLOBAL in kinds:
			return _GLOBAL

		rule = self._make_rule(node)
		if rule is None:
			return _GLOBAL

		self.rule[node] = rule
		if rule.lax is not None:
			self.lax[node] = rule.lax
			self.length[node] = rule.length
		return _LOCAL

	def _make_rule(self, node):
		if node.target is operator.getitem:
			return self._getitem(node)

		name = str(node.target)
		if name in _ELEMENTWISE:
			return self._elementwise(node)
		if name == "aten.dropout.default":
			return None if _bind(node)["train"] else self._elementwise(node)
		if name in ("aten.cat.default", "aten.stack.default"):
			return self._concat(node, name)
		if name in _CONVS:
			return self._conv(node, name)
		if name in _POOLS:
			return self._pool(node, name)
		if name in ("aten.pad.default", "aten.constant_pad_nd.default"):
			return self._pad(node)
		if name == "aten.slice.Tensor":
			return self._slice(node)
		if name == "aten.flip.default":
			return self._flip(node)
		return self._single(node, name)

	###

	def _data_lax(self, node):
		"""Return the length axis of a local input, or None."""

		if not isinstance(node, Node) or self.kind.get(node) != _LOCAL:
			return None
		return self.lax.get(node)

	def _only_varying(self, node, source):
		"""Return True when `source` is the only input that depends on X."""

		return all(n is source or self.kind[n] < _LOCAL for n in _leaves(node))

	def _elementwise(self, node):
		"""Elementwise operations, with broadcasting between their inputs."""

		out = _meta(node)
		if _batched_shape(out) is None:
			return None
		ndim = out.dim()

		tensors = [n for n in _leaves(node)
			if isinstance(_meta(n), torch.Tensor)]

		lax = None
		for n in tensors:
			if self.kind[n] != _LOCAL:
				continue
			if self.lax.get(n) is None or _meta(n).dim() != ndim:
				return None
			if lax is not None and lax != self.lax[n]:
				return None
			lax = self.lax[n]

		if lax is None:
			return None

		length = out.shape[lax]
		data = {}
		for n in tensors:
			m = _meta(n)
			if self.kind[n] == _LOCAL:
				if m.shape[lax] != length:
					return None
				data[n] = (lax, ndim)
			else:
				axis = lax - (ndim - m.dim())
				if axis >= 0 and m.shape[axis] != 1:
					data[n] = (lax, ndim)

		return _Aligned(node, lax, length, data)

	def _concat(self, node, name):
		bound = _bind(node)
		tensors = bound["tensors"]
		out = _meta(node)
		if _batched_shape(out) is None:
			return None

		laxes = {self.lax.get(n) for n in tensors if self.kind[n] == _LOCAL}
		if len(laxes) != 1 or None in laxes:
			return None
		lax = laxes.pop()

		ndim = _meta(tensors[0]).dim()
		dim = _norm(bound["dim"], ndim + (name == "aten.stack.default"))
		if dim == 0 or (name == "aten.cat.default" and dim == lax):
			return None

		out_lax = lax + (name == "aten.stack.default" and dim <= lax)
		length = out.shape[out_lax]
		if any(_meta(n).dim() != ndim or _meta(n).shape[lax] != length
			for n in tensors):
			return None

		return _Aligned(node, out_lax, length, {n: (lax, ndim)
			for n in tensors})

	def _getitem(self, node):
		source, index = node.args
		rule = self.rule.get(source)
		if rule is None or rule.element_lax is None:
			return None

		lax = rule.element_lax[index]
		return _GetItem(node, source, index, lax, _meta(node).shape[lax])

	def _conv(self, node, name):
		bound = _bind(node)
		source, weight = bound["input"], bound["weight"]
		lax = self._data_lax(source)
		if lax is None or lax < 2 or not self._only_varying(node, source):
			return None
		if bound.get("transposed", False):
			return None

		out = _meta(node)
		n = _meta(source).dim() - 2
		si = lax - 2
		kernel = list(_meta(weight).shape[2:])
		stride = _per_dim(bound["stride"], n)
		dilation = _per_dim(bound["dilation"], n)

		padding = bound["padding"]
		if isinstance(padding, str):
			if padding == "valid":
				padding = [0] * n
			else:
				totals = [d * (k - 1) for d, k in zip(dilation, kernel)]
				if any(t % 2 for i, t in enumerate(totals) if i != si):
					return None
				padding = [t // 2 for t in totals]
		else:
			padding = _per_dim(padding, n)

		inner = list(padding)
		inner[si] = 0
		groups = bound["groups"]
		bias = bound["bias"]

		if name == "aten.convolution.default":
			output_padding = bound["output_padding"]
			def call(get, width):
				return torch.ops.aten.convolution.default(get(source),
					get(weight), get(bias) if bias is not None else None,
					stride, inner, dilation, False, output_padding, groups)
		else:
			op = getattr(torch.ops.aten, "conv{}d".format(n)).default
			def call(get, width):
				return op(get(source), get(weight),
					get(bias) if bias is not None else None, stride, inner,
					dilation, groups)

		return _Window(node, source, lax, out.shape[lax], stride[si],
			dilation[si], kernel[si], padding[si], 0, call)

	def _pool(self, node, name):
		bound = _bind(node)
		source = bound["self"]
		lax = self._data_lax(source)
		if lax is None or not self._only_varying(node, source):
			return None

		n = 2 if "2d" in name else 1
		ndim = _meta(source).dim()
		first = ndim - n
		if lax < first:
			return None
		si = lax - first

		kernel = _per_dim(bound["kernel_size"], n)
		stride = _per_dim(bound["stride"] or kernel, n)
		padding = _per_dim(bound["padding"], n)
		dilation = _per_dim(bound.get("dilation", 1), n)
		ceil_mode = bound["ceil_mode"]

		in_shape, out_shape = _meta(source).shape, _meta(node).shape
		def floor_length(i):
			return (in_shape[first + i] + 2 * padding[i] - dilation[i]
				* (kernel[i] - 1) - 1) // stride[i] + 1

		# The window is computed without ceil_mode, which is exact along the
		# length axis but not along another axis with a partial last window.
		if ceil_mode and any(out_shape[first + i] != floor_length(i)
			for i in range(n) if i != si):
			return None

		inner = list(padding)
		inner[si] = 0

		if "max_pool" in name:
			op = getattr(torch.ops.aten, "max_pool{}d".format(n)).default
			def call(get, width):
				return op(get(source), kernel, stride, inner, dilation, False)
			fill = float("-inf")
		else:
			# A partial last window, or padding left out of the count, changes
			# the divisor at the edge, which the explicit fill cannot express.
			count_include_pad = bound["count_include_pad"]
			if out_shape[lax] != floor_length(si):
				return None
			if padding[si] > 0 and not count_include_pad:
				return None

			op = getattr(torch.ops.aten, "avg_pool{}d".format(n)).default
			extra = (bound["divisor_override"],) if n == 2 else ()
			def call(get, width):
				return op(get(source), kernel, stride, inner, False,
					count_include_pad, *extra)
			fill = 0

		return _Window(node, source, lax, out_shape[lax], stride[si],
			dilation[si], kernel[si], padding[si], fill, call)

	def _pad(self, node):
		bound = _bind(node)
		source = bound["self"]
		lax = self._data_lax(source)
		if lax is None or not self._only_varying(node, source):
			return None
		if bound.get("mode", "constant") != "constant":
			return None

		pad = list(bound["pad"])
		ndim = _meta(source).dim()
		if any(isinstance(p, Node) for p in pad) or len(pad) // 2 > ndim - 1:
			return None

		i = ndim - 1 - lax
		left = pad[2 * i] if 2 * i < len(pad) else 0
		inner = list(pad)
		if 2 * i < len(pad):
			inner[2 * i] = 0
			inner[2 * i + 1] = 0

		value = bound["value"]
		value = 0 if value is None else value

		def call(get, width):
			x = get(source)
			if not any(inner):
				return x
			return torch.ops.aten.constant_pad_nd.default(x, inner, value)

		return _Window(node, source, lax, _meta(node).shape[lax], 1, 1, 1,
			left, value, call)

	def _slice(self, node):
		bound = _bind(node)
		source = bound["self"]
		lax = self._data_lax(source)
		if lax is None or not self._only_varying(node, source):
			return None

		ndim = _meta(source).dim()
		dim = _norm(bound["dim"], ndim)
		start, end, step = bound["start"], bound["end"], bound["step"]
		if any(isinstance(v, Node) for v in (start, end, step)):
			return None

		if dim == 0:
			if (start or 0) == 0 and step == 1 and (end is None or end
				>= 2 ** 31):
				return self._single_output(node, source, lax)
			return None

		if dim != lax:
			return self._single_output(node, source, lax)

		length = _meta(source).shape[lax]
		start = 0 if start is None else start
		start = min(max(start + length if start < 0 else start, 0), length)

		def call(get, width):
			return torch.ops.aten.slice.Tensor(get(source), lax, 0,
				(width - 1) * step + 1, step)

		return _Window(node, source, lax, _meta(node).shape[lax], step, 1, 1,
			-start, 0, call)

	def _flip(self, node):
		bound = _bind(node)
		source = bound["self"]
		lax = self._data_lax(source)
		if lax is None:
			return None

		ndim = _meta(source).dim()
		dims = [_norm(d, ndim) for d in bound["dims"]]
		if 0 in dims:
			return None
		if lax in dims:
			return _Flip(node, source, lax, _meta(node).shape[lax], dims)
		return self._single_output(node, source, lax)

	def _single_output(self, node, source, out_lax, call=None,
		element_lax=None):
		"""Build an aligned rule for an operation with one data input."""

		value = _meta(node)
		if element_lax is None:
			if _batched_shape(value) is None:
				return None
			length = value.shape[out_lax]
		else:
			shapes = [_batched_shape(v) if lax is not None else [None]
				for v, lax in zip(value, element_lax)]
			if any(s is None for s in shapes):
				return None
			lengths = {v.shape[lax] for v, lax in zip(value, element_lax)
				if lax is not None}
			if len(lengths) != 1:
				return None
			length = lengths.pop()

			# Only the outputs that run along the length axis can be selected,
			# since the others are never computed over a window.
			for user in node.users:
				if (user.target is not operator.getitem
					or element_lax[user.args[1]] is None):
					return None

		axis, ndim = self.lax[source], _meta(source).dim()
		return _Aligned(node, out_lax, length, {source: (axis, ndim)}, call=call,
			element_lax=element_lax)

	def _single(self, node, name):
		"""Operations with one input that is read along the length axis and
		other inputs, such as weights, that are passed through."""

		if not hasattr(node.target, "_schema"):
			return None

		bound = _bind(node)
		names = [a.name for a in node.target._schema.arguments]
		data_name = "indices" if name == "aten.embedding.default" else names[0]
		source = bound[data_name]
		lax = self._data_lax(source)
		if lax is None or not self._only_varying(node, source):
			return None

		m = _meta(source)
		ndim = m.dim()

		if name in ("aten.transpose.int", "aten.permute.default"):
			if name == "aten.transpose.int":
				d0, d1 = _norm(bound["dim0"], ndim), _norm(bound["dim1"], ndim)
				dims = list(range(ndim))
				dims[d0], dims[d1] = d1, d0
			else:
				dims = [_norm(d, ndim) for d in bound["dims"]]
			if dims[0] != 0:
				return None
			return self._single_output(node, source, dims.index(lax))

		if name == "aten.unsqueeze.default":
			dim = _norm(bound["dim"], ndim + 1)
			if dim == 0:
				return None
			return self._single_output(node, source, lax + (dim <= lax))

		if name.startswith("aten.squeeze."):
			dims = bound.get("dim")
			if dims is None:
				dims = list(range(ndim))
			elif isinstance(dims, int):
				dims = [dims]
			dims = sorted({_norm(d, ndim) for d in dims
				if m.shape[_norm(d, ndim)] == 1})
			if 0 in dims or lax in dims:
				return None

			def call(get, b, width):
				return torch.ops.aten.squeeze.dims(get(source), dims)

			return self._single_output(node, source,
				lax - sum(d < lax for d in dims), call=call)

		if name == "aten.index.Tensor":
			# Indexing one axis other than the length, such as reordering the
			# channels with X[:, order].
			indices = [(d, t) for d, t in enumerate(bound["indices"])
				if t is not None]
			if len(indices) != 1 or indices[0][0] in (0, lax):
				return None
			dim, index = indices[0]
			if _meta(index).dtype == torch.bool:
				return None
			k = _meta(index).dim()
			return self._single_output(node, source, lax + (k - 1) * (dim < lax))

		if name == "aten.select.int":
			dim = _norm(bound["dim"], ndim)
			if dim in (0, lax):
				return None
			return self._single_output(node, source, lax - (dim < lax))

		if name in _RESHAPES:
			return self._reshape(node, source, lax)

		if name == "aten.expand.default":
			out = _batched_shape(_meta(node))
			if out is None or len(out) != ndim or out[lax] != m.shape[lax]:
				return None

			def call(get, b, width):
				size = [b] + out[1:]
				size[lax] = width
				return torch.ops.aten.expand.default(get(source), size)

			return self._single_output(node, source, lax, call=call)

		if name == "aten.linear.default":
			return None if lax == ndim - 1 else self._single_output(node,
				source, lax)

		if name == "aten.matmul.default":
			if _meta(bound["other"]).dim() > 2 or lax >= ndim - 1:
				return None
			return self._single_output(node, source, lax)

		if name == "aten.embedding.default":
			return self._single_output(node, source, lax)

		if name == "aten.prelu.default":
			if lax == 1 and _meta(bound["weight"]).numel() > 1:
				return None
			return self._single_output(node, source, lax)

		if name in ("aten.layer_norm.default", "aten.native_layer_norm.default",
			"aten.rms_norm.default"):
			if lax >= ndim - len(bound["normalized_shape"]):
				return None
			if name != "aten.native_layer_norm.default":
				return self._single_output(node, source, lax)
			return self._single_output(node, source, lax,
				element_lax=[lax, lax, lax])

		if name == "aten._native_batch_norm_legit_no_training.default":
			if lax == 1:
				return None
			return self._single_output(node, source, lax,
				element_lax=[lax, None, None])

		if name == "aten.batch_norm.default":
			if lax == 1 or bound["training"]:
				return None
			return self._single_output(node, source, lax)

		if name in _ALONG_DIM:
			dim = _norm(bound[_ALONG_DIM[name]], ndim)
			if dim in (0, lax):
				return None
			return self._single_output(node, source, lax)

		if name in _REDUCTIONS:
			dims = bound["dim"]
			if dims is None or (not isinstance(dims, int) and len(dims) == 0):
				return None
			dims = [dims] if isinstance(dims, int) else dims
			dims = {_norm(d, ndim) for d in dims}
			if 0 in dims or lax in dims:
				return None

			out_lax = lax if bound["keepdim"] else lax - sum(d < lax
				for d in dims)
			if name in ("aten.max.dim", "aten.min.dim"):
				return self._single_output(node, source, out_lax,
					element_lax=[out_lax, out_lax])
			return self._single_output(node, source, out_lax)

		if name in _SPLITS:
			dim = _norm(bound["dim"], ndim)
			if dim in (0, lax):
				return None
			out_lax = lax - (dim < lax) if name == "aten.unbind.int" else lax
			return self._single_output(node, source, out_lax,
				element_lax=[out_lax] * len(_meta(node)))

		return None

	def _reshape(self, node, source, lax):
		"""A reshape that keeps the batch axis and the length axis whole,
		splitting or merging only the other axes."""

		shape = _batched_shape(_meta(source))
		out = _batched_shape(_meta(node))
		if shape is None or out is None:
			return None
		if str(_meta(source).shape[0]) != str(_meta(node).shape[0]):
			return None

		pre = int(numpy.prod(shape[1:lax]))
		post = int(numpy.prod(shape[lax + 1:]))
		out_lax = None
		for j in range(1, len(out)):
			if (int(numpy.prod(out[1:j])) == pre and out[j] == shape[lax]
				and int(numpy.prod(out[j + 1:])) == post):
				out_lax = j
				break

		if out_lax is None:
			return None

		def call(get, b, width):
			size = [b] + out[1:]
			size[out_lax] = width
			return torch.ops.aten.reshape.default(get(source), size)

		return self._single_output(node, source, out_lax, call=call)

	###

	def _regions(self):
		"""Find the window of each local node that every edit can change.

		Each edit position p starts as the interval [p, p + 1) of X, and each
		rule maps its inputs' intervals to its output's. A node's window has
		the widest interval's width, clipped to the node's length, and starts
		at its interval's start for each position, shifted left where needed
		to stay inside the node.
		"""

		positions = numpy.arange(self.start, self.end)
		self.lo, self.hi = {}, {}
		self.width, self.starts = {}, {}

		for node in self.local_order:
			if node is self.x:
				lo, hi = positions, positions + 1
			else:
				lo, hi = self.rule[node].interval(self)

			self.lo[node], self.hi[node] = lo, hi

			length = self.length[node]
			nonempty = lo < hi
			width = int((hi - lo)[nonempty].max()) if nonempty.any() else 1
			width = min(max(width, 1), length)
			self.width[node] = width
			self.starts[node] = numpy.clip(numpy.where(nonempty, lo, 0), 0,
				length - width)

		self.inputs = {}
		self.starts_t = {}
		for node in self.local_order:
			self.starts_t[node] = torch.from_numpy(self.starts[node]).to(
				self.device)
			if node is self.x or isinstance(self.rule[node], _GetItem):
				continue

			rule = self.rule[node]
			reads = {}
			for source, (starts, width, fill) in rule.ranges(
				self.starts[node], self.width[node]).items():
				if rule.data[source] is None:
					axis, ndim = self.lax[source], _meta(source).dim()
				else:
					axis, ndim = rule.data[source]

				m = _meta(source)
				length = m.shape[axis - (ndim - m.dim())]
				inside = not ((starts < 0).any() or (starts + width > length).any())

				if self.kind[source] != _LOCAL:
					mode = "const"
				elif (width == self.width[source] and inside
					and numpy.array_equal(starts, self.starts[source])):
					mode = "direct"
				else:
					mode = "mixed"

				reads[source] = _Read(source, axis, ndim, starts, width, fill,
					length, mode, self.starts.get(source), self.device)

			self.inputs[node] = reads

	###

	def _static_values(self):
		"""Evaluate the nodes that depend on neither X nor the batch."""

		self.static = {}
		with torch.no_grad(), self._autocast():
			for node in self.static_order:
				if node.op == "placeholder":
					self.static[node] = self.placeholders[node.name]
				elif node.op == "get_attr":
					self.static[node] = getattr(self.ep.graph_module,
						node.target)
				else:
					self.static[node] = node.target(
						*map_arg(node.args, self.static.__getitem__),
						**map_arg(node.kwargs, self.static.__getitem__))

	def _batched_values(self, b):
		"""Evaluate the nodes that depend on `args` or the batch size, for
		the current example at batch size `b`."""

		values = {}
		args = self._example_args or []

		def get(n):
			return self.static[n] if self.kind[n] == _STATIC else values[n]

		for node in self.batched_order:
			if node.op == "placeholder":
				a = args[self.arg_nodes.index(node)]
				values[node] = a.repeat(b, *(1 for _ in a.shape[1:]))
			elif str(node.target) in _SHAPE_QUERIES and self.kind[
				node.args[0]] >= _LOCAL:
				sizes = [_concrete(d, b) for d in _meta(node.args[0]).shape]
				if str(node.target) == "aten.sym_size.int":
					values[node] = sizes[node.args[1]]
				else:
					values[node] = int(numpy.prod(sizes))
			else:
				values[node] = node.target(*map_arg(node.args, get),
					**map_arg(node.kwargs, get))

		return values

	def _set_example(self, i):
		"""Cache the activations of the local nodes for example i."""

		if self._example == i:
			return

		self._example = i
		self._example_args = None
		if self.args is not None:
			self._example_args = [self._cast(a[i:i+1]) for a in self.args]

		self._batched = {}
		values = self._batched_for(2)

		def get(n):
			kind = self.kind[n]
			if kind == _STATIC:
				return self.static[n]
			if kind == _BATCHED:
				return values[n]
			return env[n]

		# The reference runs at batch size two, so that no operation sees a
		# batch of one that its captured shape did not allow for.
		env = {self.x: self._cast(self.X[i:i+1]).repeat(2, 1, 1)}
		with torch.no_grad(), self._autocast():
			for node in self.local_order:
				if node is not self.x:
					env[node] = node.target(*map_arg(node.args, get),
						**map_arg(node.kwargs, get))

		self.ref = {n: v[:1] for n, v in env.items()
			if isinstance(v, torch.Tensor)}

		# Each read of a cached value, with its length axis moved last and
		# padded with the read's fill so that every position it reads exists.
		self.padded = {}
		for reads in self.inputs.values():
			for read in reads.values():
				kind = self.kind[read.source]
				if read.mode == "direct" or kind == _BATCHED:
					continue

				x = self.ref[read.source] if kind == _LOCAL else self.static[
					read.source]
				x = _right_align(x, read.ndim).movedim(read.axis, -1)
				self.padded[read] = torch.ops.aten.constant_pad_nd.default(x,
					[read.left, read.right], read.fill)

	def _batched_for(self, b):
		if b not in self._batched:
			with torch.no_grad(), self._autocast():
				self._batched[b] = self._batched_values(b)
		return self._batched[b]

	def _arange(self, width):
		if width not in self._aranges:
			self._aranges[width] = torch.arange(width, device=self.device)
		return self._aranges[width]

	def _gather(self, x, starts, width, b):
		"""Read `width` positions of the last axis of x, from `starts` in
		each row of the batch."""

		positions = starts[:, None] + self._arange(width)
		shape = (b,) + (1,) * (x.dim() - 2) + (width,)
		size = (b,) + tuple(x.shape[1:-1]) + (width,)
		if x.shape[0] != b:
			x = x.expand(b, *x.shape[1:])
		return torch.gather(x, -1, positions.view(shape).expand(size))

	def _materialize(self, read, idx, idx_np, b, windows, values):
		"""Read an input over the positions an output window needs.

		Positions inside the input's own window come from that window, other
		positions inside the input come from the cached reference, and
		positions outside the input are the read's fill. When every edit in
		the batch places the window at the same offset within the read, the
		read is the window with reference values joined on either side;
		otherwise each position is chosen between the two.
		"""

		if read.mode == "direct":
			return windows[read.source]

		if self.kind[read.source] == _BATCHED:
			x = _right_align(values[read.source], read.ndim).movedim(read.axis,
				-1)
			x = torch.ops.aten.constant_pad_nd.default(x,
				[read.left, read.right], read.fill)
		else:
			x = self.padded[read]

		starts = read.starts[idx]
		width = read.width

		if read.mode == "const":
			out = self._gather(x, starts, width, b)
		else:
			w = windows[read.source].movedim(read.axis, -1)
			offsets = read.offsets_np[idx_np]
			o = int(offsets[0])
			lo, hi = max(o, 0), min(o + w.shape[-1], width)

			if (offsets == o).all() and lo < hi:
				parts = [w[..., lo - o:hi - o]]
				if lo > 0:
					parts.insert(0, self._gather(x, starts, lo, b))
				if hi < width:
					parts.append(self._gather(x, starts + hi, width - hi, b))
				out = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
			else:
				out = self._gather(x, starts, width, b)
				shape = (b,) + (1,) * (out.dim() - 2) + (width,)
				w_width = w.shape[-1]
				rel = self._arange(width) - read.offsets[idx][:, None]
				inside = (rel >= 0) & (rel < w_width)
				from_window = torch.gather(w, -1, rel.clamp(0, w_width - 1).view(
					shape).expand(out.shape))
				out = torch.where(inside.view(shape), from_window, out)

		out = out.movedim(-1, read.axis)
		return out if read.axis == read.ndim - 1 else out.contiguous()

	def _splice(self, node, idx, b, windows):
		"""Write a local node's windows into copies of its full reference."""

		axis = self.lax[node]
		w = windows[node].movedim(axis, -1)
		x = self.ref[node].movedim(axis, -1).expand(b, *w.shape[1:-1],
			self.length[node]).clone()

		positions = self.starts_t[node][idx][:, None] + self._arange(w.shape[-1])
		shape = (b,) + (1,) * (w.dim() - 2) + (w.shape[-1],)
		x.scatter_(-1, positions.view(shape).expand(w.shape), w)

		x = x.movedim(-1, axis)
		return x if axis == x.dim() - 1 else x.contiguous()

	def _forward(self, chars, positions, idx_np):
		"""Run the model on one batch of edits of the current example."""

		b = chars.shape[0]
		idx = positions - self.start
		values = self._batched_for(b)

		x = torch.zeros(b, self.alphabet, 1, dtype=self.dtype,
			device=self.device)
		x[torch.arange(b, device=self.device), chars, 0] = 1
		windows = {self.x: x}

		for node in self.local_order:
			if node is self.x:
				continue

			rule = self.rule[node]
			if isinstance(rule, _GetItem):
				windows[node] = windows[rule.source][rule.index]
				continue

			reads = self.inputs[node]
			cache = {}

			def get(n):
				if n in reads:
					if n not in cache:
						cache[n] = self._materialize(reads[n], idx, idx_np, b,
							windows, values)
					return cache[n]
				return self.static[n] if self.kind[n] == _STATIC else values[n]

			windows[node] = rule.call(get, b, self.width[node])

		full = {n: self._splice(n, idx, b, windows) for n in self.frontier}
		env = {}

		def get(n):
			kind = self.kind[n]
			if kind == _STATIC:
				return self.static[n]
			if kind == _BATCHED:
				return values[n]
			if kind == _LOCAL:
				return full[n]
			return env[n]

		for node in self.global_order:
			env[node] = node.target(*map_arg(node.args, get),
				**map_arg(node.kwargs, get))

		outputs = map_arg(self.output.args[0], get)
		return tree_unflatten(tree_flatten(outputs)[0], self.out_spec)

	def predict_edits(self, i, chars, positions, batch_size, func=None):
		"""Return the model's output for single-base edits of example i.


		Parameters
		----------
		i: int
			The index of the example in `X`.

		chars: torch.Tensor, shape=(n_edits,)
			The character each edit substitutes in.

		positions: torch.Tensor, shape=(n_edits,)
			The position of each edit, within [start, end).

		batch_size: int
			The number of edits to run at a time.

		func: function or None, optional
			A function applied to the model's output for each batch, as in
			`predict`. Default is None.


		Returns
		-------
		y: torch.Tensor or list of torch.Tensors
			The output for each edit, on the CPU, formatted as `predict`
			formats it.
		"""

		self._set_example(i)
		idx_np = (positions - self.start).numpy()
		chars = chars.to(self.device)
		positions = positions.to(self.device)

		y = []
		for k in range(0, chars.shape[0], batch_size):
			c, p = chars[k:k+batch_size], positions[k:k+batch_size]
			idx_ = idx_np[k:k+batch_size]
			n = c.shape[0]

			# Runs a single edit as a batch of two, so that no operation sees a
			# batch of one that its captured shape did not allow for.
			if n == 1:
				c, p, idx_ = c.repeat(2), p.repeat(2), idx_.repeat(2)

			with torch.no_grad():
				with self._autocast():
					y_ = self._forward(c, p, idx_)

				if func is not None:
					y_ = func(y_)

			if isinstance(y_, torch.Tensor):
				y_ = y_[:n].cpu()
			elif isinstance(y_, (list, tuple)):
				y_ = tuple(yi[:n].cpu() for yi in y_)
			else:
				raise ValueError("Cannot interpret output from model.")

			y.append(y_)

		if isinstance(y[0], torch.Tensor):
			return torch.cat(y)
		return [torch.cat(y_) for y_ in zip(*y)]
