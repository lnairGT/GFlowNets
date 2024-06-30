import torch


class GFlowNet(torch.nn.Module):
	def __init__(self, tokenizer, num_hidden_states=512, num_actions=26):
		super().__init__()
		self.num_hidden_states = num_hidden_states
		self.model = torch.nn.Sequential(
			torch.nn.Linear(1, self.num_hidden_states),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(self.num_hidden_states, num_actions),
		)
		self.tokenizer = tokenizer

	def forward(self, x):
		# x --> current state
		if type(x) is not list:
			x = [x]
		w_tensors = []
		for w in x:
			w_tensors.append(self.tokenizer.encode(w))
		w_tensors = torch.Tensor(w_tensors).unsqueeze(-1)
		r = self.model(w_tensors)
		r_min = r.min()
		return r + r_min.abs()  # Try variations --> Flow must be positive
