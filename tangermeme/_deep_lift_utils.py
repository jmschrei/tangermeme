# _deep_lift_utils.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Valeh Amiri and Ruchir Rastogi

from __future__ import annotations

import math

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from numpy.polynomial.legendre import leggauss


class _HookState:
	"""A process-wide switch controlling whether the DeepLIFT hooks do work.

	A hook that needs to re-run its own module, as the local-IG rule does,
	would otherwise trip the forward hooks a second time and overwrite the
	very activations it is in the middle of reading. Flipping this switch off
	for the duration of that inner pass makes every hook a no-op without
	unregistering and re-registering it.

	The state is global rather than per-module because the hooks are plain
	functions with no shared instance to hang it on. Use `_disable_hooks`
	rather than setting this directly, so the previous value is restored even
	if the inner pass raises.
	"""

	enabled = True


@contextmanager
def _disable_hooks():
	"""Temporarily make every DeepLIFT hook a no-op.

	Restores the previous value on exit rather than unconditionally
	re-enabling, so nesting works and an inner block leaving does not switch
	the hooks back on for an outer block that is still disabled. The restore
	sits in a `finally`, so a raising inner pass cannot leave every later
	attribution silently un-hooked.


	Returns
	-------
	context: contextlib.contextmanager
		A context manager that disables the hooks for the duration of its
		block.
	"""

	previous = _HookState.enabled
	_HookState.enabled = False
	try:
		yield
	finally:
		_HookState.enabled = previous


def _hooks_disabled():
	"""Return whether the DeepLIFT hooks are currently switched off.


	Returns
	-------
	disabled: bool
		True when the hooks have been switched off by `_disable_hooks`.
	"""

	return not _HookState.enabled


def _nonlinear(module, grad_input, grad_output):
	"""An internal function implementing a general-purpose nonlinear correction.

	This function, copied and slightly modified from Captum, is meant to be
	the `rescale` rule applied to general non-linear functions such as
	activations.

	The rescale rule replaces the gradient of a non-linearity with the ratio
	of the change in its output to the change in its input, measured between
	the observed activation and the reference activation. That secant, rather
	than the tangent an ordinary backward pass would use, is what makes the
	attributions sum to the difference in model output. Where the input
	barely moves between the two the ratio is numerically unstable, so the
	plain gradient is substituted below a threshold of 1e-6.
	
	Parameters
	----------
	module: torch.nn.Module
		The module being corrected. The backward hook has put `module.input`
		and `module.output` on it for the forward call being unwound, each
		holding the observed batch concatenated with the reference batch along
		the first axis.

	grad_input: tuple of torch.tensor
		What torch would pass to a full backward hook as the gradient with
		respect to the module's inputs: the upstream gradient propagated through
		this module's ordinary local gradient. Used as the fallback wherever the
		rescale ratio is numerically unstable.

	grad_output: tuple of torch.tensor
		The upstream gradient handed to a full backward hook: the gradient of
		the model output with respect to this module's output.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The replacement gradient with respect to the module's input, i.e. the
		upstream gradient propagated through the DeepLIFT multiplier for this
		module instead of through its ordinary local gradient.
	"""

	delta_in_ = torch.sub(*module.input.chunk(2))
	delta_out_ = torch.sub(*module.output.chunk(2))

	delta_in = torch.cat([delta_in_, delta_in_])
	delta_out = torch.cat([delta_out_, delta_out_])

	delta = delta_out / delta_in
	idxs = torch.abs(delta_in) < 1e-6

	return (torch.where(idxs, grad_input[0], grad_output[0] * delta),)


def _unpool(values, indices, shape):
	"""Route pooled values back to the input positions that won each max.

	`torch.nn.functional.max_unpool1d` and its 2D counterpart assign rather
	than accumulate, so when one input position is the argmax of more than one
	output window -- which happens whenever the pooling windows overlap, as
	they do for `MaxPool1d(4, 2)` -- every contribution but one is silently
	dropped. Scatter-adding keeps all of them, which is what the rescale rule
	needs for the routed change to sum to the change in the pooled output.
	The two agree exactly when the windows do not overlap.

	Parameters
	----------
	values: torch.tensor
		The values to route back, shaped like the pooled output.

	indices: torch.tensor
		The argmax indices returned by `max_pool1d` or `max_pool2d` when
		called with `return_indices=True`, which index into the flattened
		spatial dimensions of the un-pooled input.

	shape: torch.Size or tuple
		The shape of the un-pooled input, `(batch, channels, *spatial)`.


	Returns
	-------
	unpooled: torch.tensor
		`values` accumulated into a tensor of shape `shape`.
	"""

	n, d = shape[0], shape[1]

	unpooled = torch.zeros(n, d, math.prod(shape[2:]), dtype=values.dtype,
		device=values.device)
	unpooled.scatter_add_(-1, indices.reshape(n, d, -1),
		values.reshape(n, d, -1))

	return unpooled.reshape(shape)


def _maxpool(module, grad_input, grad_output):
	"""An internal function implementing a max-pooling correction.

	This function, copied and slightly modified from Captum, is meant to be
	the `rescale` rule applied to max pooling layers given their nature of
	aggregating values across multiple positions.

	Pooling is not elementwise, so the rescale rule cannot be applied
	position by position. Instead the change in the pooled output is routed
	back through the pooling indices, which sends each output's share to
	whichever input position won the max and accumulates the shares landing
	on the same position, and the result is divided by the change in the
	input. Despite the name, both MaxPool1d and MaxPool2d are handled;
	anything else raises.
	
	Parameters
	----------
	module: torch.nn.Module
		The module being corrected. The backward hook has put `module.input`
		and `module.output` on it for the forward call being unwound, each
		holding the observed batch concatenated with the reference batch along
		the first axis.

	grad_input: tuple of torch.tensor
		What torch would pass to a full backward hook as the gradient with
		respect to the module's inputs: the upstream gradient propagated through
		this module's ordinary local gradient. Its observed half is substituted
		at the positions where the input does not change at all, which are the
		only ones without a well-defined ratio.

	grad_output: tuple of torch.tensor
		The upstream gradient handed to a full backward hook: the gradient of
		the model output with respect to this module's output.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The replacement gradient with respect to the module's input, i.e. the
		upstream gradient propagated through the DeepLIFT multiplier for this
		module instead of through its ordinary local gradient.


	Raises
	------
	ValueError
		If the module is neither a MaxPool1d nor a MaxPool2d.
	"""

	if isinstance(module, torch.nn.MaxPool1d):
		pool_func = F.max_pool1d
	elif isinstance(module, torch.nn.MaxPool2d):
		pool_func = F.max_pool2d
	else:
		raise ValueError("module must be either MaxPool1d or MaxPool2d")


	with torch.no_grad():
		delta_in_ = torch.sub(*module.input.chunk(2))

		output, output_ref = module.output.chunk(2)
		delta_out_xmax = torch.max(output, output_ref)
		delta_out = torch.cat([delta_out_xmax - output_ref, 
			output - delta_out_xmax])

		_, indices = pool_func(module.input, module.kernel_size, module.stride, 
			module.padding, module.dilation, module.ceil_mode, True)

		unpool_ = _unpool(grad_output[0] * delta_out, indices,
			module.input.shape)
		unpool_delta, unpool_ref_delta = torch.chunk(unpool_, 2)

	unpool_delta_ = unpool_delta + unpool_ref_delta

	# A position only picks up routed change when it wins a window, and that
	# change is bounded by the change in the position itself, so the quotient
	# cannot blow up however small the denominator gets and only an exact zero
	# needs guarding. Whatever is substituted there has to be a value the two
	# halves agree on, because a pooling rule further up reads both of them;
	# substituting `grad_input` whole, as this used to, does not, and that is
	# enough to stop the pair summing to delta. The multiplier is the same for
	# both halves, so it is built once at half width.
	idxs_ = delta_in_ == 0
	denominator_ = delta_in_.masked_fill(idxs_, 1)

	half = torch.where(idxs_, grad_input[0].chunk(2)[0],
		unpool_delta_ / denominator_)

	new_grad_inp = torch.cat([half, half])
	return (new_grad_inp,)


def _layer_normalization_helper(module, grad_input, grad_output,
	norm_type: str = "layernorm"):
	"""An internal function implementing the correction for the norm layers.

	LayerNorm and RMSNorm differ only in whether the input is mean-centred
	first, so both rules are derived here and the two public hooks select
	between them. Neither layer is elementwise: every output position depends
	on every input position in the normalized window, through the mean and
	the variance. A rescale rule applied position by position would therefore
	miss most of the dependence.

	Writing y_i = γ_i * a_i * v + β_i, where a_i = x_i - μ and
	v = (σ^2 + ε) ** -0.5, the multiplier from input j to output i is

		m_ji = γ_i * [ (v + v_ref)/2 * (δ_ij - 1/D)
			+ (a_i + a_ref_i)/2 * (Δv / Δσ^2) * (a_j + a_ref_j) / D ]

	where δ_ij is the Kronecker delta, equal to 1 if i = j and 0 otherwise.

	Multiplying by the upstream gradient g_i and summing over the outputs gives
	the vectorized form computed below, in two terms

		grad_in_j = (v + v_ref)/2 * (g_tilde_j - mean(g_tilde_j))   [Term 1]
			+ (Δv / Δ(σ^2)) * (a_j + a_ref_j)/(2D) * sum_i[g_tilde_i * (a_i + a_ref_i)]   [Term 2]

	where g_tilde_i = g_i * γ_i is the upstream gradient scaled by the affine
	weight.

	RMSNorm is the same expression with a_i = x_i and no mean subtraction, and
	so has no mean term in the first half.

	The ratio Δv / Δ(σ^2) is evaluated in closed form as
	-v^2 * v_ref^2 / (v + v_ref) rather than as a difference quotient, which
	avoids the cancellation that would otherwise dominate when the two
	variances are close.
	
	Parameters
	----------
	module: torch.nn.Module
		The module being corrected. The backward hook has put `module.input`
		and `module.output` on it for the forward call being unwound, each
		holding the observed batch concatenated with the reference batch along
		the first axis.

	grad_input: tuple of torch.tensor
		What torch would pass to a full backward hook as the gradient with
		respect to the module's inputs: the upstream gradient propagated through
		this module's ordinary local gradient. Unused; the closed form is
		evaluated everywhere and needs no fallback.

	grad_output: tuple of torch.tensor
		The upstream gradient handed to a full backward hook: the gradient of
		the model output with respect to this module's output.

	norm_type: str, optional
		Either "layernorm", which mean-centres the input first, or "rmsnorm",
		which does not. Default is "layernorm".


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The replacement gradient with respect to the module's input, i.e. the
		upstream gradient propagated through the DeepLIFT multiplier for this
		module instead of through its ordinary local gradient.
	"""

	assert norm_type in ["layernorm", "rmsnorm"], (
		f"Unsupported norm_type: {norm_type}")

	x, x_ref = module.input.chunk(2)

	# Normalized dimensions: last len(normalized_shape) dims
	n_norm_dims = len(module.normalized_shape)
	D = 1
	for s in module.normalized_shape:
		D *= s
	norm_dims = list(range(-n_norm_dims, 0))

	# Mean-centered inputs: a_i = x_i - mu
	if norm_type == "rmsnorm":
		a = x
		a_ref = x_ref
	else:
		mu = x.mean(dim=norm_dims, keepdim=True)
		mu_ref = x_ref.mean(dim=norm_dims, keepdim=True)
		a = x - mu
		a_ref = x_ref - mu_ref

	# Inverse std: v = (σ^2 + eps)^{-1/2}
	var = (a ** 2).mean(dim=norm_dims, keepdim=True)
	var_ref = (a_ref ** 2).mean(dim=norm_dims, keepdim=True)
	# `torch.nn.RMSNorm.eps` defaults to None, which `F.rms_norm` interprets
	# as the dtype's epsilon; mirror that here so the hook matches the forward
	# pass it is correcting.
	eps = module.eps if module.eps is not None else torch.finfo(x.dtype).eps
	v = (var + eps) ** (-0.5)
	v_ref = (var_ref + eps) ** (-0.5)

	# gamma-scaled upstream gradients for both halves of the batch
	gamma = module.weight if module.weight is not None else 1.0
	g_tilde, g_tilde_ref = (grad_output[0] * gamma).chunk(2)

	# some additional terms
	v_avg = (v + v_ref) / 2
	a_sum = a + a_ref

	# ratio = Δv / Δ(σ^2) 
	#       = (v - v_ref) / (v^{-2} - v_ref^{-2})
	#       = -v^2 * v_ref^2 / (v + v_ref)
	#       = -v^2 * v_ref^2 / (2 * v_avg)
	ratio = (-(v ** 2) * (v_ref ** 2)) / (2 * v_avg)

	def _compute_grad(g_tilde_):
		if norm_type == "rmsnorm":
			term1 = v_avg * g_tilde_
		else:
			term1 = v_avg * (g_tilde_ - g_tilde_.mean(dim=norm_dims, keepdim=True))
		dot = (g_tilde_ * a_sum).sum(dim=norm_dims, keepdim=True)
		term2 = ratio * a_sum / (2 * D) * dot
		return term1 + term2

	grad_in = _compute_grad(g_tilde)
	grad_in_ref = _compute_grad(g_tilde_ref)

	return (torch.cat([grad_in, grad_in_ref]),)


def _layernorm(module, grad_input, grad_output):
	"""An internal function implementing the correction for LayerNorm.

	A thin wrapper around `_layer_normalization_helper`, which carries the
	derivation for both norm layers. LayerNorm mean-centres the input before
	scaling it, so the multiplier picks up a mean term that RMSNorm does not
	have.


	Parameters
	----------
	module: torch.nn.LayerNorm
		The module being corrected. The backward hook has put `module.input`
		and `module.output` on it for the forward call being unwound, each
		holding the observed batch concatenated with the reference batch along
		the first axis.

	grad_input: tuple of torch.tensor
		What torch would pass to a full backward hook as the gradient with
		respect to the module's inputs: the upstream gradient propagated through
		this module's ordinary local gradient. Unused; the closed form is
		evaluated everywhere and needs no fallback.

	grad_output: tuple of torch.tensor
		The upstream gradient handed to a full backward hook: the gradient of
		the model output with respect to this module's output.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The replacement gradient with respect to the module's input, i.e. the
		upstream gradient propagated through the DeepLIFT multiplier for this
		module instead of through its ordinary local gradient.
	"""

	return _layer_normalization_helper(module, grad_input, grad_output,
		norm_type="layernorm")


def _rmsnorm(module, grad_input, grad_output):
	"""An internal function implementing the correction for RMSNorm.

	A thin wrapper around `_layer_normalization_helper`, which carries the
	derivation for both norm layers. RMSNorm skips the mean subtraction, so
	the same expression is evaluated with a_i = x_i and without the mean term.

	`torch.nn.RMSNorm.eps` defaults to None, which the forward pass reads as
	the dtype's epsilon; the helper mirrors that so the rule matches the
	forward pass it is correcting.


	Parameters
	----------
	module: torch.nn.RMSNorm
		The module being corrected. The backward hook has put `module.input`
		and `module.output` on it for the forward call being unwound, each
		holding the observed batch concatenated with the reference batch along
		the first axis.

	grad_input: tuple of torch.tensor
		What torch would pass to a full backward hook as the gradient with
		respect to the module's inputs: the upstream gradient propagated through
		this module's ordinary local gradient. Unused; the closed form is
		evaluated everywhere and needs no fallback.

	grad_output: tuple of torch.tensor
		The upstream gradient handed to a full backward hook: the gradient of
		the model output with respect to this module's output.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The replacement gradient with respect to the module's input, i.e. the
		upstream gradient propagated through the DeepLIFT multiplier for this
		module instead of through its ordinary local gradient.
	"""

	return _layer_normalization_helper(module, grad_input, grad_output,
		norm_type="rmsnorm")


def _bilinear(module, grad_input, grad_output):
	"""An internal function implementing the correction for bilinear tensor
	products using the DeepLIFT midpoint product rule.

	For a scalar product y = ab the rule gives the multipliers

		m_{a -> y} = (b + b_ref) / 2
		m_{b -> y} = (a + a_ref) / 2

	which are the ordinary derivatives of y at the midpoint of a straight-line
	path from the reference values to the observed ones. The same holds for a
	bilinear tensor product, whose multiplier matrix is the ordinary Jacobian
	evaluated at that midpoint. `torch.autograd.grad` can therefore produce the
	Jacobian-vector product against the upstream gradient directly, without
	materializing the Jacobian.

	Unlike the other rules here this returns two multipliers rather than one,
	because the module takes two operands.


	Parameters
	----------
	module: BilinearOp
		The module being corrected, carrying the two cached operands as
		`module.left` and `module.right` and the contraction as
		`module.equation`. Each operand holds the observed batch concatenated
		with the reference batch along the first axis.

	grad_input: tuple of torch.tensor
		What torch would pass to a full backward hook as the gradients with
		respect to the module's inputs: the upstream gradient propagated through
		this module's ordinary local gradient. Unused; the rule is well defined
		everywhere and needs no fallback.

	grad_output: tuple of torch.tensor
		The upstream gradient handed to a full backward hook: the gradient of
		the model output with respect to this module's output.


	Returns
	-------
	grad_input: tuple of two torch.tensor
		The replacement gradients with respect to the left and right operands,
		i.e. the upstream gradient propagated through the DeepLIFT multiplier for
		each operand instead of through this module's ordinary local gradient.
	"""

	left, left_ref = module.left.chunk(2)
	right, right_ref = module.right.chunk(2)
	grad_out, grad_out_ref = grad_output[0].chunk(2)

	left_mid = (0.5 * (left + left_ref)).detach().requires_grad_(True)
	right_mid = (0.5 * (right + right_ref)).detach().requires_grad_(True)

	with torch.enable_grad():
		if module.equation is None:
			out = torch.matmul(left_mid, right_mid)
		elif module.equation == "...,...->...":
			out = left_mid * right_mid
		else:
			out = torch.einsum(module.equation, left_mid, right_mid)

		grad_in_left, grad_in_right = torch.autograd.grad(
			out,
			(left_mid, right_mid),
			grad_outputs=grad_out,
			retain_graph=True,
			create_graph=False,
		)

		grad_in_left_ref, grad_in_right_ref = torch.autograd.grad(
			out,
			(left_mid, right_mid),
			grad_outputs=grad_out_ref,
			retain_graph=False,
			create_graph=False,
		)

	return (
		torch.cat([grad_in_left, grad_in_left_ref], dim=0),
		torch.cat([grad_in_right, grad_in_right_ref], dim=0),
	)


def _softmax(module, grad_input, grad_output):
	"""An internal function implementing the correction for softmax.

	This rule decomposes the operation into steps that each have an exact
	multiplier and then chains them

		a_i = exp(x_i - c),   s = sum_i a_i,   v = 1 / s,   y_i = a_i * v

	where c is subtracted from both the observed and the reference logits for
	numerical stability and cancels out of the result.

	The rule is applied along whichever axis the module normalizes over. Only
	the batch axis is rejected, because DeepLIFT stacks each example with its
	reference along it and a softmax over that axis would mix the two.


	Parameters
	----------
	module: torch.nn.Softmax
		The module being corrected, carrying the cached `module.input` and the
		axis to normalize over as `module.dim`.

	grad_input: tuple of torch.tensor
		What torch would pass to a full backward hook as the gradient with
		respect to the module's inputs: the upstream gradient propagated through
		this module's ordinary local gradient. Unused.

	grad_output: tuple of torch.tensor
		The upstream gradient handed to a full backward hook: the gradient of
		the model output with respect to this module's output.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The replacement gradient with respect to the module's input, i.e. the
		upstream gradient propagated through the DeepLIFT multiplier for this
		module instead of through its ordinary local gradient.


	Raises
	------
	ValueError
		If the module normalizes over the batch axis.
	"""

	dim = module.dim
	if dim < 0:
		dim = module.input.ndim + dim

	if dim == 0:
		raise ValueError("Softmax over the batch axis is not supported. "
			"DeepLIFT stacks each example with its reference along that axis, "
			"so normalizing over it would mix the two.")

	x, x_ref = module.input.chunk(2, dim=0)
	grad_out, grad_out_ref = grad_output[0].chunk(2, dim=0)

	# Picking c = max(x, x_ref) makes the largest exponent exactly exp(0)=1,
	# so nothing overflows.
	c = torch.maximum(
		x.max(dim=dim, keepdim=True).values,
		x_ref.max(dim=dim, keepdim=True).values,
	)
	a = torch.exp(x - c)
	a_ref = torch.exp(x_ref - c)
	s = a.sum(dim=dim, keepdim=True)
	s_ref = a_ref.sum(dim=dim, keepdim=True)
	# Guard against s underflowing to exactly zero, which would make the
	# reciprocal +inf and reach log(v) below as NaN. It takes a whole softmax
	# window sitting ~88 (fp32) below the shared max c, so only a mask that
	# differs between an example and its reference gets there. Such a row
	# carries no attribution mass, so the clamped value itself does not matter.
	tiny = torch.finfo(s.dtype).tiny
	v = s.clamp_min(tiny).reciprocal()
	v_ref = s_ref.clamp_min(tiny).reciprocal()
	y = a * v
	y_ref = a_ref * v_ref

	# Multiplier for x_j -> a_j, with derivative fallback.
	# m_{x_j -> a_j} = Δa_j / Δx_j
	delta_x = x - x_ref
	mult_x_to_a = torch.where(delta_x.abs() > 1e-6, (a - a_ref) / delta_x,
		a_ref)

	# Define a few intermediate quantities for the log-ratio multipliers.
	delta_y = y - y_ref
	delta_v = v - v_ref
	delta_log_a = x - x_ref
	delta_log_v = torch.log(v) - torch.log(v_ref)
	delta_log_y = delta_log_a + delta_log_v

	delta_y_over_delta_log_y = torch.where(
		delta_log_y.abs() > 1e-6,
		delta_y / delta_log_y,
		y_ref,
	)
	delta_log_v_over_delta_v = torch.where(
		delta_v.abs() > 1e-6,
		delta_log_v / delta_v,
		v_ref.reciprocal(),
	)

	# Log-ratio multiplier for the denominator path (m_{v -> y_i}).
	#  m_{v -> y_i} = (Δy_i / Δlog(y_i)) * (Δlog(v) / Δv)
	mult_v_to_y = delta_y_over_delta_log_y * delta_log_v_over_delta_v

	# Combine the a_i -> y_i path with the v -> y_i path. Also multiply by the upstream gradient (grad_out) to
	# derive the downstream gradient (grad_in).
	#
	#     grad_in_j = grad_out_j * (Δy_j / Δlog(y_j))
	#         - m_{x_j -> a_j} * sum_i grad_out_i
	#             * (1 / (s * s_ref))
	#             * (Δy_i / Δlog(y_i))
	#             * (Δlog(v) / Δv)
	#
	# The numerator path carries no factor of m_{x_j -> a_j}, because
	# m_{x_j -> a_j} * (Δlog(a_j) / Δa_j) is Δlog(a_j) / Δx_j, which is one:
	# log(a_j) is x_j - c, and the same c is subtracted from both sides.
	# Guarding the two ratios separately is what breaks on a peaked softmax,
	# where Δa trips its threshold while Δx does not and the Δlog(a)/Δa
	# fallback returns 1/a_ref. Folding them out also removes the 0 * inf a
	# fully masked logit used to produce, since no reciprocal of a_ref is
	# taken at all.
	reciprocal_mult = -v * v_ref # reuse the clamped reciprocals
	grad_in = (
		grad_out * delta_y_over_delta_log_y
		+ mult_x_to_a
			* (grad_out * mult_v_to_y * reciprocal_mult).sum(dim=dim,
				keepdim=True)
	)
	grad_in_ref = (
		grad_out_ref * delta_y_over_delta_log_y
		+ mult_x_to_a
			* (grad_out_ref * mult_v_to_y * reciprocal_mult).sum(dim=dim,
				keepdim=True)
	)

	return (torch.cat([grad_in, grad_in_ref], dim=0),)


def _gauss_legendre(n_points):
	"""Return Gauss-Legendre nodes and weights mapped from [-1, 1] to [0, 1].

	Gauss-Legendre quadrature is defined on [-1, 1], but the path integral it
	approximates here runs from the reference activation to the observed one,
	parameterized over [0, 1]. Both the nodes and the weights are rescaled
	once, when the hook is built, rather than on every backward pass.


	Parameters
	----------
	n_points: int
		The number of quadrature points.


	Returns
	-------
	alphas: numpy.ndarray, shape=(n_points,)
		The quadrature nodes, on [0, 1].

	weights: numpy.ndarray, shape=(n_points,)
		The corresponding weights, summing to one.
	"""

	nodes, weights = leggauss(n_points)
	# Map [-1, 1] onto [0, 1], the path z(t) = z0 + t * (z - z0) runs over.
	alphas = (nodes + 1.0) / 2.0
	weights = weights / 2.0
	return alphas, weights
