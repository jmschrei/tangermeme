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


class SharedReluModel(torch.nn.Module):
	def __init__(self):
		super(SharedReluModel, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 8, (5,))
		self.conv2 = torch.nn.Conv1d(8, 4, (3,))
		self.shared_relu = torch.nn.ReLU()
		self.flatten = torch.nn.Flatten()
		self.linear = torch.nn.Linear(376, 1)

	def forward(self, X):
		X = self.shared_relu(self.conv1(X))
		X = self.shared_relu(self.conv2(X))
		X = self.flatten(X)
		return self.linear(X)


class MultipleSharedActivationsModel(torch.nn.Module):
	def __init__(self):
		super(MultipleSharedActivationsModel, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 8, (5,), padding='same')
		self.conv2 = torch.nn.Conv1d(8, 8, (3,), padding='same')
		self.conv3 = torch.nn.Conv1d(8, 4, (3,), padding='same')
		self.shared_relu = torch.nn.ReLU()
		self.shared_tanh = torch.nn.Tanh()
		self.flatten = torch.nn.Flatten()
		self.linear = torch.nn.Linear(400, 1)

	def forward(self, X):
		X = self.shared_relu(self.conv1(X))
		X = self.shared_tanh(X)
		X = self.shared_relu(self.conv2(X))
		X = self.shared_tanh(X)
		X = self.shared_relu(self.conv3(X))
		X = self.flatten(X)
		return self.linear(X)


class SharedPoolModel(torch.nn.Module):
	def __init__(self):
		super(SharedPoolModel, self).__init__()
		self.conv1 = torch.nn.Conv1d(4, 8, (3,), padding='same')
		self.conv2 = torch.nn.Conv1d(8, 8, (3,), padding='same')
		self.shared_relu = torch.nn.ReLU()
		self.shared_pool = torch.nn.MaxPool1d(2)
		self.flatten = torch.nn.Flatten()
		self.linear = torch.nn.Linear(200, 1)

	def forward(self, X):
		X = self.shared_pool(self.shared_relu(self.conv1(X)))
		X = self.shared_pool(self.shared_relu(self.conv2(X)))
		X = self.flatten(X)
		return self.linear(X)
