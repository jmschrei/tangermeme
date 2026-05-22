# toy_models.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


class SumModel(torch.nn.Module):
	def __init__(self):
		super(SumModel, self).__init__()
		
	def forward(self, X):
		return X.sum(axis=-1)


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
	"""Three Conv1d layers with dilation 1, 2, 4, padding='same'."""

	def __init__(self, seq_len=100, n_outputs=1):
		super(DilatedConv, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 8, (3,), dilation=1, padding='same')
		self.conv2 = torch.nn.Conv1d(8, 8, (3,), dilation=2, padding='same')
		self.conv3 = torch.nn.Conv1d(8, 8, (3,), dilation=4, padding='same')
		self.relu = torch.nn.ReLU()
		self.dense = torch.nn.Linear(8 * seq_len, n_outputs)

	def forward(self, X):
		h = self.relu(self.conv1(X))
		h = self.relu(self.conv2(h))
		h = self.relu(self.conv3(h))
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

	def forward(self, X):
		h = self.relu(self.ln(self.conv(X)))
		return self.dense(h.reshape(h.shape[0], -1))


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
