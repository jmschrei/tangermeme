# toy_models.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from tangermeme.deep_lift_shap import BilinearOp
from tangermeme._deep_lift_utils import _hooks_disabled

torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


class SumModel(torch.nn.Module):
	def __init__(self):
		super(SumModel, self).__init__()
		
	def forward(self, X):
		return X.sum(axis=-1)


class SoftmaxModel(torch.nn.Module):
	def __init__(self):
		super(SoftmaxModel, self).__init__()
		self.softmax = torch.nn.Softmax(dim=-1)

	def forward(self, x):
		return self.softmax(x)


class FlattenDense(torch.nn.Module):
	def __init__(self, seq_len=100, n_outputs=3):
		super(FlattenDense, self).__init__()
		self.dense = torch.nn.Linear(seq_len*4, n_outputs)
		self.seq_len = seq_len

	def forward(self, X, alpha=0, beta=1):
		X = X.reshape(X.shape[0], self.seq_len*4)
		return self.dense(X) * beta + alpha


class Conv(torch.nn.Module):
	def __init__(self):
		super(Conv, self).__init__()
		self.conv = torch.nn.Conv1d(4, 12, (3,))

	def forward(self, X):
		return self.conv(X)


class Conv1(torch.nn.Module):
	def __init__(self):
		super(Conv1, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 12, (3,))
		self.conv2 = torch.nn.Conv1d(12, 1, (5,))

	def forward(self, X):
		return self.conv2(self.conv1(X))[:, 0]


class Scatter(torch.nn.Module):
	def __init__(self):
		super(Scatter, self).__init__()

	def forward(self, X):
		return X.permute(0, 2, 1)


class ConvDense(torch.nn.Module):
	def __init__(self, n_outputs=3):
		super(ConvDense, self).__init__()

		self.dense = torch.nn.Linear(400, n_outputs)
		self.conv = torch.nn.Conv1d(4, 12, (3,))

	def forward(self, X, alpha=0):
		return self.conv(X) + alpha, self.dense(X.reshape(X.shape[0], -1)) 


class ConvAvgDense(torch.nn.Module):
	def __init__(self, n_outputs=1):
		super(ConvAvgDense, self).__init__()

		self.conv = torch.nn.Conv1d(4, 12, (3,))
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(12, n_outputs)

	def forward(self, X):
		return self.dense(self.relu(self.conv(X)).mean(dim=-1))


class ConvPoolDense(torch.nn.Module):
	def __init__(self):
		super(ConvPoolDense, self).__init__()

		self.conv1 = torch.nn.Conv1d(4, 32, (3,), padding='same')
		self.pool1 = torch.nn.MaxPool1d(3)

		self.conv2 = torch.nn.Conv1d(32, 16, (5,), padding='same')
		self.pool2 = torch.nn.MaxPool1d(3)

		self.flatten = torch.nn.Flatten()
		self.dense = torch.nn.Linear(176, 1)

		self.relu1 = torch.nn.ReLU()
		self.relu2 = torch.nn.ReLU()

	def forward(self, X):
		X = self.pool1(self.relu1(self.conv1(X)))
		X = self.pool2(self.relu2(self.conv2(X)))
		y = self.dense(self.flatten(X))
		return y


class SmallDeepSEA(torch.nn.Module):
	def __init__(self, n_outputs=1):
		super(SmallDeepSEA, self).__init__()

		self.conv1 = torch.nn.Conv1d(4, 32, (3,), padding='same')
		self.pool1 = torch.nn.MaxPool1d(3)
		self.relu1 = torch.nn.ReLU()

		self.conv2 = torch.nn.Conv1d(32, 16, (3,), padding='same')
		self.pool2 = torch.nn.MaxPool1d(3)
		self.relu2 = torch.nn.ReLU()

		self.linear1 = torch.nn.Linear(176, 20)
		self.relu3 = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(20, n_outputs)

	def forward(self, X):
		X = self.relu1(self.pool1(self.conv1(X)))
		X = self.relu2(self.pool2(self.conv2(X)))
		X = X.reshape(X.shape[0], -1)
		X = self.relu3(self.linear1(X))
		return self.linear2(X)


class ResidualConv(torch.nn.Module):
	"""Residual block: act(conv1(X)) -> conv2 -> + X -> dense.

	Channel count is kept at 4 throughout the residual stream so the skip add
	is shape-compatible. The activation is a constructor argument so the same
	class is reused for the activation sweep in test_deep_lift_shap.py.
	"""

	def __init__(self, seq_len=100, n_outputs=1, activation=torch.nn.ReLU):
		super(ResidualConv, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 4, (3,), padding='same')
		self.act = activation()
		self.conv2 = torch.nn.Conv1d(4, 4, (3,), padding='same')
		self.dense = torch.nn.Linear(4 * seq_len, n_outputs)

	def forward(self, X):
		h = self.act(self.conv1(X))
		h = self.conv2(h) + X
		return self.dense(h.reshape(h.shape[0], -1))


class Transformer(torch.nn.Module):
	"""Permute -> single TransformerEncoderLayer -> permute -> dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(Transformer, self).__init__()
		layer = torch.nn.TransformerEncoderLayer(d_model=4, nhead=2,
			dim_feedforward=8, batch_first=True)
		self.encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
		self.dense = torch.nn.Linear(4 * seq_len, n_outputs)

	def forward(self, X):
		h = X.permute(0, 2, 1)
		h = self.encoder(h)
		h = h.permute(0, 2, 1)
		return self.dense(h.reshape(h.shape[0], -1))


class ConvRuleSeq(torch.nn.Module):
	"""conv -> one newly registered rule -> conv, keeping the length axis.

	`pisa` attributes every output position separately, so it needs a model
	that still has a length axis at its output rather than one that flattens
	down to a scalar the way the models above do. `rule` selects which of the
	operations added to the rule table sits in the middle.
	"""

	def __init__(self, rule="layernorm", seq_len=15, channels=8):
		super(ConvRuleSeq, self).__init__()
		self.rule = rule
		self.conv = torch.nn.Conv1d(4, channels, (3,), padding='same')
		self.out = torch.nn.Conv1d(channels, 1, (3,))

		if rule == "layernorm":
			self.op = torch.nn.LayerNorm([channels, seq_len])
		elif rule == "rmsnorm":
			self.op = torch.nn.RMSNorm([channels, seq_len])
		elif rule == "softmax":
			self.op = torch.nn.Softmax(dim=-1)
		elif rule == "bilinear":
			self.op = BilinearOp("...,...->...")
			self.gate = torch.nn.Conv1d(4, channels, (3,), padding='same')
		else:
			raise ValueError("Unknown rule: {}".format(rule))

	def forward(self, X):
		h = self.conv(X)

		if self.rule == "bilinear":
			h = self.op(h, self.gate(X))
		else:
			h = self.op(h)

		# `pisa` indexes the output as (example, position), so the channel axis
		# is squeezed out the same way Conv1 does it.
		return self.out(h)[:, 0]


class MultiHeadAttention(torch.nn.Module):
	"""Multi-head self-attention built only from modules DeepLIFT can hook.

	`torch.nn.MultiheadAttention` computes its softmax and both of its matmuls
	functionally, so there is no module for a rule to attach to and the whole
	attention block is silently treated as linear. Writing the same
	computation with `BilinearOp` and `torch.nn.Softmax` puts every
	non-linearity behind a module, which is what makes a transformer block
	attributable. The scaling sits outside the op, as BilinearOp's contract
	asks.
	"""

	def __init__(self, seq_len=100, d_model=8, n_heads=2, n_outputs=1):
		super(MultiHeadAttention, self).__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = d_model // n_heads

		self.proj = torch.nn.Conv1d(4, d_model, (3,), padding='same')
		self.q = torch.nn.Linear(d_model, d_model)
		self.k = torch.nn.Linear(d_model, d_model)
		self.v = torch.nn.Linear(d_model, d_model)

		self.scores = BilinearOp("nhld,nhmd->nhlm")
		self.softmax = torch.nn.Softmax(dim=-1)
		self.context = BilinearOp("nhlm,nhmd->nhld")

		self.norm = torch.nn.LayerNorm([seq_len, d_model])
		self.dense = torch.nn.Linear(seq_len * d_model, n_outputs)

	def _split(self, h):
		n, length, _ = h.shape
		h = h.reshape(n, length, self.n_heads, self.head_dim)
		return h.permute(0, 2, 1, 3)

	def forward(self, X, alpha=0, beta=1):
		h = self.proj(X).permute(0, 2, 1)
		q, k, v = self._split(self.q(h)), self._split(self.k(h)), self._split(self.v(h))

		a = self.softmax(self.scores(q, k) / (self.head_dim ** 0.5))
		c = self.context(a, v).permute(0, 2, 1, 3)
		c = c.reshape(h.shape[0], h.shape[1], self.d_model)

		h = self.norm(c + h)
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class TransformerBlock(torch.nn.Module):
	"""A stackable transformer block built only from modules DeepLIFT can hook.

	`MultiHeadAttention` is the attention operation on its own. This is the
	block a user actually stacks around it: attention, a residual, two norms
	and a feedforward with a non-linearity. The arrangement matters to the
	rules rather than only to the model, because pre-norm and post-norm put
	the LayerNorm on opposite sides of the residual add, so the norm rule sees
	a different graph in each and `n_blocks` chains every rule through the
	output of the block below it.

	A learned positional embedding is a parameter added to the stream, which
	is linear and needs no rule of its own; it is here to confirm that.


	Parameters
	----------
	seq_len: int, optional
		The length of the input sequences. Default is 100.

	d_model: int, optional
		The width of the residual stream. Default is 8.

	n_heads: int, optional
		The number of attention heads, which must divide `d_model`. Default
		is 2.

	n_blocks: int, optional
		The number of blocks to stack. Default is 1.

	pre_norm: bool, optional
		Whether to normalize the input of each sublayer and leave the residual
		path clear (`True`), or normalize the sum afterwards (`False`).
		Default is False.

	activation: str, optional
		Either "gelu" or "silu", the non-linearity in the feedforward.
		Default is "gelu".

	positional: bool, optional
		Whether to add a learned positional embedding after the input
		projection. Default is False.

	mask: torch.tensor or None, optional
		An additive mask, broadcast onto the attention scores before the
		softmax. If None, no mask is applied. Default is None.

	n_outputs: int, optional
		The number of outputs. Default is 1.
	"""

	def __init__(self, seq_len=100, d_model=8, n_heads=2, n_blocks=1,
			pre_norm=False, activation="gelu", positional=False, mask=None,
			n_outputs=1):
		super(TransformerBlock, self).__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = d_model // n_heads
		self.n_blocks = n_blocks
		self.pre_norm = pre_norm

		self.proj = torch.nn.Conv1d(4, d_model, (3,), padding='same')

		self.pos = None
		if positional:
			self.pos = torch.nn.Parameter(
				torch.randn(1, seq_len, d_model) * 0.1)

		# Registered even when None, so that `self.mask` exists either way and
		# a real mask follows the model onto whatever device it is moved to.
		self.register_buffer("mask", mask)

		def _stack(fn):
			return torch.nn.ModuleList([fn() for i in range(n_blocks)])

		self.q = _stack(lambda: torch.nn.Linear(d_model, d_model))
		self.k = _stack(lambda: torch.nn.Linear(d_model, d_model))
		self.v = _stack(lambda: torch.nn.Linear(d_model, d_model))

		self.scores = _stack(lambda: BilinearOp("nhld,nhmd->nhlm"))
		self.softmax = _stack(lambda: torch.nn.Softmax(dim=-1))
		self.context = _stack(lambda: BilinearOp("nhlm,nhmd->nhld"))

		self.norm1 = _stack(lambda: torch.nn.LayerNorm([seq_len, d_model]))
		self.norm2 = _stack(lambda: torch.nn.LayerNorm([seq_len, d_model]))

		act = {"gelu": torch.nn.GELU, "silu": torch.nn.SiLU}[activation]
		self.ff = _stack(lambda: torch.nn.Sequential(
			torch.nn.Linear(d_model, 2 * d_model),
			act(),
			torch.nn.Linear(2 * d_model, d_model)))

		self.dense = torch.nn.Linear(seq_len * d_model, n_outputs)

	def _split(self, h):
		n, length, _ = h.shape
		h = h.reshape(n, length, self.n_heads, self.head_dim)
		return h.permute(0, 2, 1, 3)

	def _attend(self, h, i):
		q = self._split(self.q[i](h))
		k = self._split(self.k[i](h))
		v = self._split(self.v[i](h))

		# The scaling sits outside the op, as BilinearOp's contract asks.
		s = self.scores[i](q, k) / (self.head_dim ** 0.5)
		if self.mask is not None:
			s = s + self.mask

		c = self.context[i](self.softmax[i](s), v).permute(0, 2, 1, 3)
		return c.reshape(h.shape[0], h.shape[1], self.d_model)

	def forward(self, X, alpha=0, beta=1):
		h = self.proj(X).permute(0, 2, 1)
		if self.pos is not None:
			h = h + self.pos

		for i in range(self.n_blocks):
			if self.pre_norm:
				h = h + self._attend(self.norm1[i](h), i)
				h = h + self.ff[i](self.norm2[i](h))
			else:
				h = self.norm1[i](h + self._attend(h, i))
				h = self.norm2[i](h + self.ff[i](h))

		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class AttentionPool(torch.nn.Module):
	"""Attention pooling, the layer Enformer pools with.

	Attention pooling takes a weighted average within each pool window rather
	than the max or the mean: a 1x1 convolution scores the entries of the
	window, a softmax turns those scores into weights, and the pooled value
	is the window multiplied by its weights and summed. `enformer_pytorch`
	writes both the softmax and the product as function calls, so neither has
	a module for a rule to attach to and DeepLIFT silently treats a
	non-linear layer as linear.

	`hookable` selects between the two forms. When True the softmax sits
	behind `torch.nn.Softmax` and the product behind `BilinearOp`, which is
	what makes the layer attributable; when False the functional form is
	reproduced, and with it the missing rules. The functional form can still
	be attributed by registering this class with `integrated_gradients_op`,
	which needs no rule for the operations inside it.

	Unlike the rules' other users this layer changes the length of what it is
	handed, and pads when the length is not a multiple of `pool_size`. The
	padded entries are masked out of the softmax, which is the part of the
	arrangement worth its own test, since the mask is applied outside the
	softmax module and so has to be in the activations the rule reads.


	Parameters
	----------
	channels: int, optional
		The number of channels being pooled. Default is 8.

	pool_size: int, optional
		The number of positions averaged into each output position. Default
		is 2.

	hookable: bool, optional
		Whether to write the softmax and the product as modules (`True`) or
		as function calls (`False`). Default is True.
	"""

	def __init__(self, channels=8, pool_size=2, hookable=True):
		super(AttentionPool, self).__init__()
		self.pool_size = pool_size
		self.hookable = hookable

		# Enformer's initialization: a Dirac 1x1 convolution scaled by two, so
		# that the layer begins as a soft maximum over each window instead of
		# at an arbitrary set of weights.
		self.logits = torch.nn.Conv2d(channels, channels, 1, bias=False)
		torch.nn.init.dirac_(self.logits.weight)
		with torch.no_grad():
			self.logits.weight.mul_(2)

		self.softmax = torch.nn.Softmax(dim=-1)
		self.prod = BilinearOp("...,...->...")

	def _window(self, X):
		n, channels, length = X.shape
		return X.reshape(n, channels, length // self.pool_size, self.pool_size)

	def forward(self, X):
		n, _, length = X.shape
		remainder = length % self.pool_size

		if remainder > 0:
			pad = self.pool_size - remainder
			X = torch.nn.functional.pad(X, (0, pad), value=0)

			mask = torch.zeros((n, 1, length), dtype=torch.bool,
				device=X.device)
			mask = self._window(torch.nn.functional.pad(mask, (0, pad),
				value=True))

		X = self._window(X)
		logits = self.logits(X)

		if remainder > 0:
			logits = logits.masked_fill(mask, -torch.finfo(logits.dtype).max)

		if self.hookable:
			return self.prod(X, self.softmax(logits)).sum(dim=-1)
		return (X * logits.softmax(dim=-1)).sum(dim=-1)


class ConvAttentionPool(torch.nn.Module):
	"""conv -> attention pooling -> dense.

	The model `AttentionPool` is tested through. `pool_size=3` on the default
	length of 100 leaves a remainder, which is how the layer's padded and
	masked path is reached.
	"""

	def __init__(self, seq_len=100, n_outputs=1, channels=8, pool_size=2,
			hookable=True):
		super(ConvAttentionPool, self).__init__()
		self.conv = torch.nn.Conv1d(4, channels, (3,), padding='same')
		self.relu = torch.nn.ReLU()
		self.pool = AttentionPool(channels=channels, pool_size=pool_size,
			hookable=hookable)
		self.dense = torch.nn.Linear(
			channels * -(-seq_len // pool_size), n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.pool(self.relu(self.conv(X)))
		h = self.dense(h.reshape(h.shape[0], -1))
		return h * beta + alpha


class Conv2DExpand(torch.nn.Module):
	"""Unsqueeze the alphabet axis, Conv2d that collapses height, then dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(Conv2DExpand, self).__init__()
		# Kernel height equals input height (4), so output height is 1.
		self.conv = torch.nn.Conv2d(1, 8, kernel_size=(4, 3))
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(8 * (seq_len - 2), n_outputs)

	def forward(self, X):
		h = X.unsqueeze(1)
		h = self.relu(self.conv(h))
		h = h.squeeze(2)
		return self.dense(h.reshape(h.shape[0], -1))


class _CustomLinearFunction(torch.autograd.Function):
	"""Linear scale by 2 with matching gradient. No registration needed."""

	@staticmethod
	def forward(ctx, X):
		return X * 2.0

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output * 2.0


class CustomLinear(torch.nn.Module):
	"""Model whose first op is a linear torch.autograd.Function."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(CustomLinear, self).__init__()
		self.conv = torch.nn.Conv1d(4, 4, (3,), padding='same')
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(4 * seq_len, n_outputs)

	def forward(self, X):
		h = _CustomLinearFunction.apply(X)
		h = self.relu(self.conv(h))
		return self.dense(h.reshape(h.shape[0], -1))


class _CustomSqrtFunction(torch.autograd.Function):
	"""Nonlinear sqrt with explicit backward. Inputs must be > 0."""

	@staticmethod
	def forward(ctx, X):
		Y = torch.sqrt(X)
		ctx.save_for_backward(Y)
		return Y

	@staticmethod
	def backward(ctx, grad_output):
		Y, = ctx.saved_tensors
		return grad_output * 0.5 / Y


class CustomSqrtModule(torch.nn.Module):
	"""nn.Module wrapper around the sqrt Function so DeepLIFT can hook it."""

	def forward(self, X):
		return _CustomSqrtFunction.apply(X)


class CustomSqrt(torch.nn.Module):
	"""Model whose nonlinearity is a registered custom sqrt op."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(CustomSqrt, self).__init__()
		self.conv = torch.nn.Conv1d(4, 4, (3,), padding='same')
		self.sqrt = CustomSqrtModule()
		self.dense = torch.nn.Linear(4 * seq_len, n_outputs)

	def forward(self, X):
		# Bias the conv output positive before sqrt.
		h = self.conv(X) + 5.0
		h = self.sqrt(h)
		return self.dense(h.reshape(h.shape[0], -1))


class DilatedConv(torch.nn.Module):
	"""Three Conv1d layers with dilation 1, 2, 4, padding='same'.

	One `ReLU` is called after each convolution. `share=True` calls a single
	instance three times, which is how the model is usually written and the
	reason it exercises a module used more than once in a forward pass;
	`share=False` spells the same function out with three instances.
	"""

	def __init__(self, seq_len=100, n_outputs=1, share=True):
		super(DilatedConv, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 8, (3,), dilation=1, padding='same')
		self.conv2 = torch.nn.Conv1d(8, 8, (3,), dilation=2, padding='same')
		self.conv3 = torch.nn.Conv1d(8, 8, (3,), dilation=4, padding='same')
		self.relu = torch.nn.ReLU()
		self.relu2 = self.relu if share else torch.nn.ReLU()
		self.relu3 = self.relu if share else torch.nn.ReLU()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X):
		h = self.relu(self.conv1(X))
		h = self.relu2(self.conv2(h))
		h = self.relu3(self.conv3(h))
		return self.dense(h.reshape(h.shape[0], -1))


class ConvBatchNorm(torch.nn.Module):
	"""conv -> BatchNorm1d -> relu -> dense. BN running stats are set to
	non-standard values but kept untrained so eval-mode output is deterministic.
	"""

	def __init__(self, seq_len=100, n_outputs=1):
		super(ConvBatchNorm, self).__init__()
		self.conv = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.bn = torch.nn.BatchNorm1d(8)
		# Non-standard untrained running stats so BN actually scales/shifts.
		with torch.no_grad():
			self.bn.running_mean.fill_(0.5)
			self.bn.running_var.fill_(2.0)
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X):
		h = self.relu(self.bn(self.conv(X)))
		return self.dense(h.reshape(h.shape[0], -1))


class ConvLayerNorm(torch.nn.Module):
	"""conv -> LayerNorm over (C, L) -> relu -> dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(ConvLayerNorm, self).__init__()
		self.conv = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.ln = torch.nn.LayerNorm([8, seq_len])
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.relu(self.ln(self.conv(X)))
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class ConvRMSNorm(torch.nn.Module):
	"""conv -> RMSNorm over (C, L) -> relu -> dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(ConvRMSNorm, self).__init__()
		self.conv = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.norm = torch.nn.RMSNorm([8, seq_len])
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.relu(self.norm(self.conv(X)))
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class ConvSoftmax(torch.nn.Module):
	"""conv -> softmax over a chosen axis -> dense.

	`dim` selects which axis is normalized: -1 for the length axis, 1 for the
	channel axis. The rule is applied along whichever one the module carries,
	so both are worth exercising.

	`logit_scale` multiplies the logits before the softmax. Raising it peaks
	the distribution, which drives most of the exponentials to values small
	enough that the rule's guarded ratios have to be right about them; a
	default of 1.0 leaves every other user of this model unchanged.
	"""

	def __init__(self, seq_len=100, n_outputs=1, dim=-1, channels=8,
			logit_scale=1.0):
		super(ConvSoftmax, self).__init__()
		self.dim = dim
		self.logit_scale = logit_scale
		self.conv = torch.nn.Conv1d(4, channels, (3,), padding='same')
		self.softmax = torch.nn.Softmax(dim=dim)
		self.dense = torch.nn.Linear(channels * seq_len, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.conv(X) * self.logit_scale

		# A softmax over n entries leaves every weight near 1/n, which would push
		# the attributions below what four decimal places can resolve, so the
		# distribution is rescaled by the size of the axis it normalized.
		h = self.softmax(h) * h.shape[self.dim]
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class ConvBilinear(torch.nn.Module):
	"""conv, conv -> elementwise BilinearOp product -> dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(ConvBilinear, self).__init__()
		self.left = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.right = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.op = BilinearOp("...,...->...")
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.op(self.left(X), self.right(X))
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class ConvBilinearMatmul(torch.nn.Module):
	"""conv, conv -> BilinearOp matmul into a channel Gram matrix -> dense.

	The transpose happens outside the op, because BilinearOp uses its operands
	exactly as passed. Contracting the length axis keeps the product (C, C)
	rather than (L, L), which would dominate memory at 30 shuffles.
	"""

	def __init__(self, seq_len=100, n_outputs=1):
		super(ConvBilinearMatmul, self).__init__()
		self.left = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.right = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.op = BilinearOp(None)
		self.dense = torch.nn.Linear(8 * 8, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.op(self.left(X), self.right(X).transpose(1, 2))
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class ConvBilinearEinsum(torch.nn.Module):
	"""conv, conv -> BilinearOp einsum into a channel Gram matrix -> dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(ConvBilinearEinsum, self).__init__()
		self.left = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.right = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.op = BilinearOp("ncl,ndl->ncd")
		self.dense = torch.nn.Linear(8 * 8, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.op(self.left(X), self.right(X))
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class CustomGate(torch.nn.Module):
	"""A user-defined bilinear op that is absent from the default rule table.

	It satisfies the contract `_bilinear` relies on -- an `equation` attribute
	and the two operands cached on the module -- without subclassing
	BilinearOp, so it only attributes correctly when handed to
	`additional_nonlinear_ops`. Subclassing would not work: hooks are
	registered by `isinstance` but dispatched by exact `type`.
	"""

	def __init__(self):
		super(CustomGate, self).__init__()
		self.equation = "...,...->..."

	def forward(self, left, right):
		if hasattr(self, "_NON_LINEAR_OPS") and not _hooks_disabled():
			self.left = left.detach()
			self.right = right.detach()

		return left * right


class ConvCustomGate(torch.nn.Module):
	"""conv, conv -> unregistered CustomGate -> dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(ConvCustomGate, self).__init__()
		self.left = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.right = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.op = CustomGate()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.op(self.left(X), self.right(X))
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class ScaledTanhModule(torch.nn.Module):
	"""An elementwise nonlinearity with no entry in the default rule table."""

	def forward(self, X):
		return torch.tanh(X) * 2.0 + 0.5


class ConvScaledTanh(torch.nn.Module):
	"""conv -> unregistered elementwise nonlinearity -> dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(ConvScaledTanh, self).__init__()
		self.conv = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.act = ScaledTanhModule()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		h = self.act(self.conv(X))
		return self.dense(h.reshape(h.shape[0], -1)) * beta + alpha


class MultiActivation(torch.nn.Module):
	"""conv -> GELU -> conv -> SiLU -> conv -> Tanh -> dense."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(MultiActivation, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.act1 = torch.nn.GELU()
		self.conv2 = torch.nn.Conv1d(8, 8, (3,), padding='same')
		self.act2 = torch.nn.SiLU()
		self.conv3 = torch.nn.Conv1d(8, 8, (3,), padding='same')
		self.act3 = torch.nn.Tanh()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X):
		h = self.act1(self.conv1(X))
		h = self.act2(self.conv2(h))
		h = self.act3(self.conv3(h))
		return self.dense(h.reshape(h.shape[0], -1))


class DropoutConv(torch.nn.Module):
	"""conv -> Dropout(0.5) -> relu -> dense. Dropout must be a no-op in eval."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(DropoutConv, self).__init__()
		self.conv = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.dropout = torch.nn.Dropout(p=0.5)
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X):
		h = self.relu(self.dropout(self.conv(X)))
		return self.dense(h.reshape(h.shape[0], -1))


class MultiInputMultiOutput(torch.nn.Module):
	"""Tuple-output model that also takes (alpha, beta) args.

	`alpha` shifts the conv branch; `beta` scales the dense branch. Both are
	broadcast against per-example shape (B, 1).
	"""

	def __init__(self, n_outputs=3):
		super(MultiInputMultiOutput, self).__init__()
		self.conv = torch.nn.Conv1d(4, 12, (3,))
		self.dense = torch.nn.Linear(400, n_outputs)

	def forward(self, X, alpha=0, beta=1):
		# alpha is shape (B, 1); broadcast to (B, 1, 1) against (B, 12, L-2).
		a = alpha.unsqueeze(-1) if torch.is_tensor(alpha) else alpha
		return (
			self.conv(X) + a,
			self.dense(X.reshape(X.shape[0], -1)) * beta,
		)


class AttributeNameConv(torch.nn.Module):
	"""conv -> relu -> dense, carrying plain attributes named like the caches.

	The forward hooks cache activations on non-linear modules under those two
	names, BilinearOp caches its operands under `left`/`right`, and
	`_clear_hooks` is applied to every module in the model rather than only the
	hooked ones. This model puts ordinary, non-tensor attributes of all four
	names on modules that never get hooked, so that clearing the caches can be
	checked not to take them along with it.
	"""

	def __init__(self, seq_len=100, n_outputs=1):
		super(AttributeNameConv, self).__init__()
		self.conv = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

		self.input = "sequence"
		self.output = n_outputs
		self.conv.input = "kernel"
		self.conv.output = "logits"

		# BilinearOp caches its two operands under these names, and those caches
		# are cleared across every module too.
		self.left = "5-prime"
		self.right = "3-prime"
		self.conv.left = "upstream"
		self.conv.right = "downstream"

	def forward(self, X):
		h = self.relu(self.conv(X))
		return self.dense(h.reshape(h.shape[0], -1))


class SharedActivation(torch.nn.Module):
	"""Two convolutions whose activation is one module called twice, or two.

	A module called more than once in a forward pass sees a different input
	and output each time, and a rule needs the pair belonging to the call the
	backward pass is unwinding. The two convolutions change the number of
	channels, so the two calls differ in shape as well as in value and pairing
	them up wrongly cannot go unnoticed. `share=False` builds the same
	function out of two instances, which is the answer the shared version has
	to match.
	"""

	def __init__(self, share=True, seq_len=100, n_outputs=1):
		super(SharedActivation, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 8, (5,))
		self.conv2 = torch.nn.Conv1d(8, 4, (3,))
		self.relu1 = torch.nn.ReLU()
		self.relu2 = self.relu1 if share else torch.nn.ReLU()
		self.dense = torch.nn.Linear(4 * (seq_len - 6), n_outputs)

	def forward(self, X):
		h = self.relu1(self.conv1(X))
		h = self.relu2(self.conv2(h))
		return self.dense(h.reshape(h.shape[0], -1))


class SharedPool(torch.nn.Module):
	"""Two convolutions whose max-pool is one module called twice, or two.

	The max-pool rule reads the cached input to recover the pooling indices,
	so it needs the right call's activations for a different reason than the
	elementwise rules do. The two calls see different lengths.
	"""

	def __init__(self, share=True, seq_len=100, n_outputs=1):
		super(SharedPool, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.conv2 = torch.nn.Conv1d(8, 8, (3,), padding='same')
		self.relu1 = torch.nn.ReLU()
		self.relu2 = torch.nn.ReLU()
		self.pool1 = torch.nn.MaxPool1d(2)
		self.pool2 = self.pool1 if share else torch.nn.MaxPool1d(2)
		self.dense = torch.nn.Linear(8 * (seq_len // 4), n_outputs)

	def forward(self, X):
		h = self.pool1(self.relu1(self.conv1(X)))
		h = self.pool2(self.relu2(self.conv2(h)))
		return self.dense(h.reshape(h.shape[0], -1))


class SharedBranch(torch.nn.Module):
	"""One activation used on two parallel branches rather than in sequence.

	The two calls are siblings in the graph instead of one being nested inside
	the other, so the backward pass does not reach them along a single chain.
	Pairing a call with its activations by position in the forward order, or
	by the reverse of it, gets this model wrong.
	"""

	def __init__(self, share=True, seq_len=100, n_outputs=1):
		super(SharedBranch, self).__init__()
		self.conv_a = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.conv_b = torch.nn.Conv1d(4, 6, (3,), padding='same')
		self.relu_a = torch.nn.ReLU()
		self.relu_b = self.relu_a if share else torch.nn.ReLU()
		self.dense = torch.nn.Linear(14 * seq_len, n_outputs)

	def forward(self, X):
		a = self.relu_a(self.conv_a(X))
		b = self.relu_b(self.conv_b(X))
		h = torch.cat([a, b], dim=1)
		return self.dense(h.reshape(h.shape[0], -1))


class SharedRuleSeq(torch.nn.Module):
	"""`ConvRuleSeq` with its operation applied twice, as one module or two.

	Every rule reads activations cached by the forward hooks, and `BilinearOp`
	caches its two operands itself, so each of them has to be given the values
	from the right call. The length axis is kept so that `pisa` can use this
	model as well as `deep_lift_shap`.
	"""

	def __init__(self, rule="layernorm", share=True, seq_len=15, channels=8):
		super(SharedRuleSeq, self).__init__()
		self.rule = rule
		self.conv = torch.nn.Conv1d(4, channels, (3,), padding='same')
		self.mid = torch.nn.Conv1d(channels, channels, (3,), padding='same')
		self.out = torch.nn.Conv1d(channels, 1, (3,))

		def op():
			if rule == "layernorm":
				return torch.nn.LayerNorm([channels, seq_len])
			elif rule == "rmsnorm":
				return torch.nn.RMSNorm([channels, seq_len])
			elif rule == "softmax":
				return torch.nn.Softmax(dim=-1)
			elif rule == "bilinear":
				return BilinearOp("...,...->...")
			raise ValueError("Unknown rule: {}".format(rule))

		self.op1 = op()
		self.op2 = self.op1 if share else op()

		# LayerNorm and RMSNorm initialize to a weight of one and a bias of
		# zero, which two separate instances share by accident. Moving them
		# off the defaults is what makes the shared case a genuinely
		# weight-shared layer rather than two parameterless ones.
		with torch.no_grad():
			for parameter in self.op1.parameters():
				parameter.normal_(mean=1.0, std=0.1)

		if rule == "bilinear":
			self.gate = torch.nn.Conv1d(4, channels, (3,), padding='same')

	def forward(self, X):
		h = self.conv(X)

		if self.rule == "bilinear":
			gate = self.gate(X)
			h = self.op1(h, gate)
			h = self.op2(self.mid(h), gate)
		else:
			h = self.op1(h)
			h = self.op2(self.mid(h))

		# `pisa` indexes the output as (example, position), so the channel axis
		# is squeezed out the same way ConvRuleSeq does it.
		return self.out(h)[:, 0]
