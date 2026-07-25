import torch
import torch.nn.functional as F
from contextlib import contextmanager

class HookState:
    """Global switch controlling whether the DeepLIFT hooks do work."""
    enabled = True

@contextmanager
def _disable_hooks():
    """Temporarily make every DeepLIFT hook a no-op."""

    previous = HookState.enabled
    HookState.enabled = False
    try:
        yield
    finally:
        HookState.enabled = previous

def _hooks_disabled():
    """Return True if DeepLIFT hooks are currently disabled."""
    return not HookState.enabled

def _nonlinear(module, grad_input, grad_output):
	"""An internal function implementing a general-purpose nonlinear correction.

	This function, copied and slightly modified from Captum, is meant to be
	the `rescale` rule applied to general non-linear functions such as
	activations.
	"""

	delta_in_ = torch.sub(*module.input.chunk(2))
	delta_out_ = torch.sub(*module.output.chunk(2))

	delta_in = torch.cat([delta_in_, delta_in_])
	delta_out = torch.cat([delta_out_, delta_out_])

	delta = delta_out / delta_in
	idxs = torch.abs(delta_in) < 1e-6

	return (torch.where(idxs, grad_input[0], grad_output[0] * delta),)

def _maxpool(module, grad_input, grad_output):
	"""An internal function implementing a 1D max-pooling correction.

	This function, copied and slightly modified from Captum, is meant to be
	the `rescale` rule applied to max pooling layers given their nature of
	aggregating values across multiple positions.
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

def _layernorm(module, grad_input, grad_output):
    """An internal function implementing the DeepLIFT correction for LayerNorm.

    Given y_i = gamma_i * A_i * v + beta_i
    where A_i = x_i - mu and v = (sigma^2 + eps)^{-1/2}, the DeepLIFT multiplier
    from input j to output i is:

        m_ji = gamma_i * [ (v + v0)/2 * (delta_ij - 1/D)
                         + (A_i + A0_i)/2 * (delta_v / delta_sigma2) * (A_j + A0_j) / D ]

    Multiplying by upstream gradient g_i and summing over i gives the vectorized rule:

        grad_in_j = (v + v0)/2 * (g_tilde_j - mean(g_tilde))       [Term 1]
                  + (delta_v / delta_sigma2) * (A_j + A0_j) / (2*D)
                    * sum_i( g_tilde_i * (A_i + A0_i) )             [Term 2]

    where g_tilde_i = g_i * gamma_i. When delta_sigma2 is near zero (i.e., x and
    x0 have the same variance), we fall back to the standard autograd gradient.
    """

    x, x0 = module.input.chunk(2)

    # Normalized dimensions: last len(normalized_shape) dims
    n_norm_dims = len(module.normalized_shape)
    D = 1
    for s in module.normalized_shape:
        D *= s
    norm_dims = list(range(-n_norm_dims, 0))

    # Mean-centered inputs: A_i = x_i - mu
    mu = x.mean(dim=norm_dims, keepdim=True)
    mu0 = x0.mean(dim=norm_dims, keepdim=True)
    A = x - mu
    A0 = x0 - mu0

    # Inverse std: v = (sigma^2 + eps)^{-1/2}
    var = (A ** 2).mean(dim=norm_dims, keepdim=True)
    var0 = (A0 ** 2).mean(dim=norm_dims, keepdim=True)
    v = (var + module.eps) ** (-0.5)
    v0 = (var0 + module.eps) ** (-0.5)

    delta_v = v - v0
    delta_sigma2 = var - var0

    # gamma-scaled upstream gradients for both halves of the batch
    gamma = module.weight if module.weight is not None else 1.0
    g_tilde, g0_tilde = (grad_output[0] * gamma).chunk(2)

    # Shared terms
    v_avg = (v + v0) / 2
    A_sum = A + A0

    # ratio = delta_v / delta_sigma2
    #       = (v - v0) / (v^{-2} - v0^{-2})
    #       = -v^2 * v0^2 / (v + v0)
    #       = -v^2 * v0^2 / (2 * v_avg)
    ratio = (-(v ** 2) * (v0 ** 2)) / (2 * v_avg)

    def _compute_grad(g_tilde_):
        term1 = v_avg * (g_tilde_ - g_tilde_.mean(dim=norm_dims, keepdim=True))
        dot = (g_tilde_ * A_sum).sum(dim=norm_dims, keepdim=True)
        term2 = ratio * A_sum / (2 * D) * dot
        return term1 + term2

    grad_in = _compute_grad(g_tilde)
    grad_in0 = _compute_grad(g0_tilde)

    return (torch.cat([grad_in, grad_in0]),)

def _gauss_legendre(n_points):
    """Return Gauss-Legendre nodes and weights mapped from [-1, 1] to [0, 1]."""

    from numpy.polynomial.legendre import leggauss
    print(f"_gauss_legendre n_points: {n_points}")
    nodes, weights = leggauss(n_points)
	# Map from [-1, 1] to [0, 1] for integration along the path z(t) = z0 + t*(z - z0)
    alphas = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    return alphas, weights

def make_local_ig_autograd(K=8, name=None):
    """Return a generic local-IG hook using autograd VJPs.

    This function implements integrated-gradients for any generic module. It integrates
    ``J_f(z(t))^T q`` along the local linear path from reference activation ``z0`` to
    actual activation ``z`` using Gauss-Legendre quadrature for integral approximation.

    The implementation computes VJPs, not full Jacobians. For efficiency, all
    quadrature nodes and both upstream-gradient halves are packed into one
    autograd call.

    Args:
        K (int): Number of Gauss-Legendre quadrature points. Higher K incurs more
		autograd calls but yields more accurate IG approximations.
        name (str or None): Optional name suffix for the returned hook.

    Returns:
        A hook function compatible with ``additional_nonlinear_ops``.
    """

    _nodes_list, _weights_list = _gauss_legendre(K)

    def _hook(module, grad_input, grad_output):
        dtype, device = grad_output[0].dtype, grad_output[0].device
        GL_NODES = torch.tensor(_nodes_list, dtype=dtype, device=device)
        GL_WEIGHTS = torch.tensor(_weights_list, dtype=dtype, device=device)

        z, z0 = module.input.chunk(2)
        q, q0 = grad_output[0].chunk(2)
        delta_z = z - z0
        batch_size = z.shape[0]

        z_path = torch.cat([z0 + t_k * delta_z for t_k in GL_NODES], dim=0)
        z_eval = torch.cat([z_path, z_path], dim=0).detach().requires_grad_()

        q_path = torch.cat([w_k * q.detach() for w_k in GL_WEIGHTS], dim=0)
        q0_path = torch.cat([w_k * q0.detach() for w_k in GL_WEIGHTS], dim=0)
        q_eval = torch.cat([q_path, q0_path], dim=0)

        with torch.enable_grad(), _disable_hooks():
            y_eval = module(z_eval)
            grad_eval = torch.autograd.grad(
                y_eval,
                z_eval,
                grad_outputs=q_eval,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        grad, grad0 = grad_eval.chunk(2)
        grad = grad.reshape(K, batch_size, *z.shape[1:])
        grad0 = grad0.reshape(K, batch_size, *z0.shape[1:])
        return (torch.cat([grad.sum(dim=0), grad0.sum(dim=0)]),)

    suffix = name if name is not None else "fn"
    _hook.__name__ = f"_local_ig_autograd_{suffix}_K{K}"
    return _hook

def _softmax(module, grad_input, grad_output):
    """DeepLIFT rule for softmax that respects full input-output dependencies and uses
    a log-ratio product rule.
    Softmax decomposition:
        a_i = exp(x_i - c)              c is a constant for numerical stability
        d = sum_i a_i
        r = 1 / d
        y_k = a_k * r
    """

    dim = module.dim
    if dim < 0:
        dim = module.input.ndim + dim
    if dim != module.input.ndim - 1:
        raise ValueError("Only softmax over the last dimension is currently supported.")

    x, x_ref = module.input.chunk(2, dim=0)
    gout, gout_ref = grad_output[0].chunk(2, dim=0)

    c = torch.maximum(
        x.max(dim=-1, keepdim=True).values,
        x_ref.max(dim=-1, keepdim=True).values,
    )
    a = torch.exp(x - c)
    a_ref = torch.exp(x_ref - c)
    d = a.sum(dim=-1, keepdim=True)
    d_ref = a_ref.sum(dim=-1, keepdim=True)
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
    delta_log_a_over_delta_a = torch.where(
        delta_a.abs() > 1e-6,
        delta_log_a / delta_a,
        a_ref.reciprocal(),
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
        + (gout * mult_r_to_y * reciprocal_mult).sum(dim=-1, keepdim=True)
    )
    gin_ref = mult_x_to_a * (
        gout_ref * mult_a_to_y
        + (gout_ref * mult_r_to_y * reciprocal_mult).sum(dim=-1, keepdim=True)
    )

    return (torch.cat([gin, gin_ref], dim=0),)

def _bilinear(module, grad_input, grad_output):
    """DeepLIFT symmetric product rule for bilinear operations."""

    left, left_ref = module.left.chunk(2)
    right, right_ref = module.right.chunk(2)
    gout, gout_ref = grad_output[0].chunk(2)

    left_mid = (0.5 * (left + left_ref)).detach().requires_grad_(True)
    right_mid = (0.5 * (right + right_ref)).detach().requires_grad_(True)

    with torch.enable_grad():
        if module.equation is None:
            out = torch.matmul(left_mid, right_mid)
        else:
            out = torch.einsum(module.equation, left_mid, right_mid)

        gin_left, gin_right = torch.autograd.grad(
            out,
            (left_mid, right_mid),
            grad_outputs=gout,
            retain_graph=True,
            create_graph=False,
        )

        gin_left_ref, gin_right_ref = torch.autograd.grad(
            out,
            (left_mid, right_mid),
            grad_outputs=gout_ref,
            retain_graph=False,
            create_graph=False,
        )

    return (
        torch.cat([gin_left, gin_left_ref], dim=0),
        torch.cat([gin_right, gin_right_ref], dim=0),
    )