# _deep_lift_utils.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

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

	Both the forward hooks and `BilinearOp.forward` check this before
	caching, so a re-entrant call does not clobber the cached activations.


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
		The module being corrected. The forward hooks have already cached
		`module.input` and `module.output` on it, each holding the observed
		batch concatenated with the reference batch along the first axis.

	grad_input: tuple of torch.tensor
		The gradients with respect to the module's inputs, as torch passes
		them to a full backward hook. Used as the fallback wherever the rule
		is numerically unstable.

	grad_output: tuple of torch.tensor
		The gradients with respect to the module's outputs, as torch passes
		them to a full backward hook.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The corrected gradient with respect to the module's input, which is
		the DeepLIFT multiplier for this operation.
	"""

	delta_in_ = torch.sub(*module.input.chunk(2))
	delta_out_ = torch.sub(*module.output.chunk(2))

	delta_in = torch.cat([delta_in_, delta_in_])
	delta_out = torch.cat([delta_out_, delta_out_])

	delta = delta_out / delta_in
	idxs = torch.abs(delta_in) < 1e-6

	return (torch.where(idxs, grad_input[0], grad_output[0] * delta),)


def _maxpool(module, grad_input, grad_output):
	"""An internal function implementing a max-pooling correction.

	This function, copied and slightly modified from Captum, is meant to be
	the `rescale` rule applied to max pooling layers given their nature of
	aggregating values across multiple positions.

	Pooling is not elementwise, so the rescale rule cannot be applied
	position by position. Instead the change in the pooled output is routed
	back through the pooling indices with an unpool, which sends each
	output's share to whichever input position won the max, and the result is
	divided by the change in the input. Despite the name, both MaxPool1d and
	MaxPool2d are handled; anything else raises.
	
	Parameters
	----------
	module: torch.nn.Module
		The module being corrected. The forward hooks have already cached
		`module.input` and `module.output` on it, each holding the observed
		batch concatenated with the reference batch along the first axis.

	grad_input: tuple of torch.tensor
		The gradients with respect to the module's inputs, as torch passes
		them to a full backward hook. Used as the fallback wherever the rule
		is numerically unstable.

	grad_output: tuple of torch.tensor
		The gradients with respect to the module's outputs, as torch passes
		them to a full backward hook.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The corrected gradient with respect to the module's input, which is
		the DeepLIFT multiplier for this operation.


	Raises
	------
	ValueError
		If the module is neither a MaxPool1d nor a MaxPool2d.
	"""

	if isinstance(module, torch.nn.MaxPool1d):
		pool_func, unpool_func = F.max_pool1d, F.max_unpool1d
	elif isinstance(module, torch.nn.MaxPool2d):
		pool_func, unpool_func = F.max_pool2d, F.max_unpool2d
	else:
		raise ValueError("module must be either MaxPool1d or MaxPool2d")


	with torch.no_grad():
		delta_in_ = torch.sub(*module.input.chunk(2))
		delta_in = torch.cat([delta_in_, delta_in_])

		output, output_ref = module.output.chunk(2)
		delta_out_xmax = torch.max(output, output_ref)
		delta_out = torch.cat([delta_out_xmax - output_ref, 
			output - delta_out_xmax])

		_, indices = pool_func(module.input, module.kernel_size, module.stride, 
			module.padding, module.dilation, module.ceil_mode, True)

		unpool_ = unpool_func(grad_output[0] * delta_out, indices, 
			module.kernel_size, module.stride, module.padding, 
			list(module.input.shape))
		unpool_delta, unpool_ref_delta = torch.chunk(unpool_, 2)

	unpool_delta_ = unpool_delta + unpool_ref_delta
	unpool_delta = torch.cat([unpool_delta_, unpool_delta_])
	idxs = torch.abs(delta_in) < 1e-7

	new_grad_inp = torch.where(idxs, grad_input[0], unpool_delta / delta_in)
	return (new_grad_inp,)


def _layer_normalization_helper(module, grad_input, grad_output,
	norm_type: str = "layernorm"):
	"""An internal function implementing the correction for the norm layers.

	LayerNorm and RMSNorm differ only in whether the input is mean-centred
	first, so both rules are derived here and the two public hooks select
	between them. Neither layer is elementwise: every output position depends
	on every input position in the normalized window, through the mean and
	the variance. A rescale rule applied position by position would therefore
	miss most of the dependence, which is why these need a closed form rather
	than the generic correction.

	Writing y_i = g_i * a_i * v + b_i, where a_i = x_i - mu and
	v = (var + eps) ** -0.5, the multiplier from input j to output i is

		m_ji = g_i * [ (v + v_ref)/2 * (d_ij - 1/D)
			+ (a_i + a_ref_i)/2 * (dv / dvar) * (a_j + a_ref_j) / D ]

	Multiplying by the upstream gradient and summing over the outputs gives
	the vectorized form actually computed below, in two terms

		grad_in_j = (v + v_ref)/2 * (gt_j - mean(gt))
			+ (dv / dvar) * (a_j + a_ref_j)/(2D) * sum_i[gt_i * (a_i + a_ref_i)]

	where gt_i is the upstream gradient scaled by the affine weight. RMSNorm
	is the same expression with a_i = x_i, no mean subtraction, and so no
	mean term in the first half.

	The ratio dv/dvar is evaluated in closed form as
	-v**2 * v_ref**2 / (v + v_ref) rather than as a difference quotient, which
	avoids the cancellation that would otherwise dominate when the two
	variances are close.
	
	Parameters
	----------
	module: torch.nn.Module
		The module being corrected. The forward hooks have already cached
		`module.input` and `module.output` on it, each holding the observed
		batch concatenated with the reference batch along the first axis.

	grad_input: tuple of torch.tensor
		The gradients with respect to the module's inputs, as torch passes
		them to a full backward hook. Used as the fallback wherever the rule
		is numerically unstable.

	grad_output: tuple of torch.tensor
		The gradients with respect to the module's outputs, as torch passes
		them to a full backward hook.

	norm_type: str, optional
		Either "layernorm", which mean-centres the input first, or "rmsnorm",
		which does not. Default is "layernorm".


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The corrected gradient with respect to the module's input, which is
		the DeepLIFT multiplier for this operation.
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
	# A None eps is the dtype's epsilon, matching `F.rms_norm`.
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
		The module being corrected. The forward hooks have already cached
		`module.input` and `module.output` on it, each holding the observed
		batch concatenated with the reference batch along the first axis.

	grad_input: tuple of torch.tensor
		The gradients with respect to the module's inputs, as torch passes
		them to a full backward hook. Unused; the closed form is evaluated
		everywhere and needs no fallback.

	grad_output: tuple of torch.tensor
		The gradients with respect to the module's outputs, as torch passes
		them to a full backward hook.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The corrected gradient with respect to the module's input, which is
		the DeepLIFT multiplier for this operation.
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
		The module being corrected. The forward hooks have already cached
		`module.input` and `module.output` on it, each holding the observed
		batch concatenated with the reference batch along the first axis.

	grad_input: tuple of torch.tensor
		The gradients with respect to the module's inputs, as torch passes
		them to a full backward hook. Unused; the closed form is evaluated
		everywhere and needs no fallback.

	grad_output: tuple of torch.tensor
		The gradients with respect to the module's outputs, as torch passes
		them to a full backward hook.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The corrected gradient with respect to the module's input, which is
		the DeepLIFT multiplier for this operation.
	"""

	return _layer_normalization_helper(module, grad_input, grad_output,
		norm_type="rmsnorm")


def _bilinear(module, grad_input, grad_output):
	"""An internal function implementing the correction for bilinear ops.

	A product of two quantities that both vary has no single rescale ratio,
	because the change in the output cannot be assigned to one operand or the
	other. The symmetric rule splits it evenly: each operand is credited with
	the gradient of the contraction evaluated at the midpoint between the
	observed and reference values of the other. Summed over both operands
	this reproduces the change in the output exactly, which is what keeps
	summation-to-delta intact across the layer.

	The contraction is re-run on the midpoints under `torch.enable_grad` and
	differentiated, rather than the rule being written out per equation, so
	one implementation covers matmul, einsum, and the elementwise product
	alike.

	Unlike the elementwise rules this returns two gradients rather than one,
	since the module takes two inputs.


	Parameters
	----------
	module: BilinearOp
		The module being corrected, carrying the two cached operands as
		`module.left` and `module.right` and the contraction as
		`module.equation`. Each operand holds the observed batch concatenated
		with the reference batch along the first axis.

	grad_input: tuple of torch.tensor
		The gradients with respect to the module's inputs, as torch passes
		them to a full backward hook. Unused; the rule is well defined
		everywhere and needs no fallback.

	grad_output: tuple of torch.tensor
		The gradients with respect to the module's outputs, as torch passes
		them to a full backward hook.


	Returns
	-------
	grad_input: tuple of two torch.tensor
		The corrected gradients with respect to the left and right operands.
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

	Softmax couples every output to every input through its denominator, so
	the elementwise rescale rule captures only the numerator and misses the
	dense part of the dependence entirely. This rule decomposes the operation
	into steps that each have an exact multiplier and then chains them

		a_i = exp(x_i - c),   d = sum_i a_i,   r = 1 / d,   y_k = a_k * r

	where c is subtracted from both the observed and the reference logits for
	numerical stability and cancels out of the result. The exponential and
	the reciprocal are handled in log space, where their difference quotients
	stay well conditioned, and the two paths into each output, the direct one
	through its own numerator and the shared one through the denominator, are
	summed. Each quotient falls back to the analytic derivative at the
	reference point when its denominator is below 1e-6.

	The rule is applied along whichever axis the module normalizes over. Only
	the batch axis is rejected, because DeepLIFT stacks each example with its
	reference along it and a softmax over that axis would mix the two.


	Parameters
	----------
	module: torch.nn.Softmax
		The module being corrected, carrying the cached `module.input` and the
		axis to normalize over as `module.dim`.

	grad_input: tuple of torch.tensor
		The gradients with respect to the module's inputs, as torch passes
		them to a full backward hook. Unused; every quotient in the rule has
		its own fallback.

	grad_output: tuple of torch.tensor
		The gradients with respect to the module's outputs, as torch passes
		them to a full backward hook.


	Returns
	-------
	grad_input: tuple of one torch.tensor
		The corrected gradient with respect to the module's input.


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
	gout, gout_ref = grad_output[0].chunk(2, dim=0)

	c = torch.maximum(
		x.max(dim=dim, keepdim=True).values,
		x_ref.max(dim=dim, keepdim=True).values,
	)
	a = torch.exp(x - c)
	a_ref = torch.exp(x_ref - c)
	d = a.sum(dim=dim, keepdim=True)
	d_ref = a_ref.sum(dim=dim, keepdim=True)
	r = d.reciprocal()
	r_ref = d_ref.reciprocal()
	y = a * r
	y_ref = a_ref * r_ref

	# Multiplier for x_i -> exp(x_i - c), with derivative fallback.
	delta_x = x - x_ref
	delta_a = a - a_ref
	mult_x_to_a = torch.where(delta_x.abs() > 1e-6, delta_a / delta_x, a_ref)

	# Log-ratio multipliers for the direct numerator path and denominator path.
	#     m_{a_i -> y_k} =
	#       1{i = k} * (Δy_k / Δlog(y_k)) * (Δlog(a_k) / Δa_k)
	#       - (1 / (d * d_ref)) * (Δy_k / Δlog(y_k)) * (Δlog(r) / Δr)
	delta_log_a = x - x_ref
	delta_log_r = torch.log(r) - torch.log(r_ref)
	delta_log_y = delta_log_a + delta_log_r
	delta_y = y - y_ref
	delta_r = r - r_ref

	delta_y_over_delta_log_y = torch.where(
		delta_log_y.abs() > 1e-6,
		delta_y / delta_log_y,
		y_ref,
	)
	# An attention mask drives its masked logits to a large negative value in
	# both the example and the reference, so `a` and `a_ref` underflow to
	# exactly zero there and this fallback would be infinite. `mult_x_to_a` is
	# zero at those positions, so their contribution is zero and the fallback
	# only has to be finite; leaving it infinite makes the product 0 * inf,
	# which is NaN and propagates back through every earlier layer.
	a_ref_safe = torch.where(a_ref > 0, a_ref, torch.ones_like(a_ref))
	delta_log_a_over_delta_a = torch.where(
		delta_a.abs() > 1e-6,
		delta_log_a / delta_a,
		a_ref_safe.reciprocal(),
	)
	delta_log_r_over_delta_r = torch.where(
		delta_r.abs() > 1e-6,
		delta_log_r / delta_r,
		r_ref.reciprocal(),
	)

	mult_a_to_y = delta_y_over_delta_log_y * delta_log_a_over_delta_a
	mult_r_to_y = delta_y_over_delta_log_y * delta_log_r_over_delta_r
	reciprocal_mult = -1.0 / (d * d_ref)

	# Combine the direct a_i -> y_i contribution with the dense r -> y_k path.
	#     gin_i = m_{x_i -> a_i} * [
	#         gout_i * (Δy_i / Δlog(y_i)) * (Δlog(a_i) / Δa_i)
	#         - sum_k gout_k
	#             * (1 / (d * d_ref))
	#             * (Δy_k / Δlog(y_k))
	#             * (Δlog(r) / Δr)
	#     ]
	gin = mult_x_to_a * (
		gout * mult_a_to_y
		+ (gout * mult_r_to_y * reciprocal_mult).sum(dim=dim, keepdim=True)
	)
	gin_ref = mult_x_to_a * (
		gout_ref * mult_a_to_y
		+ (gout_ref * mult_r_to_y * reciprocal_mult).sum(dim=dim, keepdim=True)
	)

	return (torch.cat([gin, gin_ref], dim=0),)


def _gauss_legendre(n_points):
	"""Return Gauss-Legendre nodes and weights mapped from [-1, 1] to [0, 1].

	Gauss-Legendre quadrature is defined on [-1, 1], but the path integral it
	is used for here runs from the reference activation to the observed one,
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
