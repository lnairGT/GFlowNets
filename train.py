import torch
from model import GFlowNet
from reward import Tokenizer, RepeatedLettersReward
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math


def check_generations(generations, reward_func, check_rep=False):
	num_good_generations = 0
	rep_letters = 0
	for word in generations:
		reward = reward_func.get_reward(word)
		if reward != 0:
			num_good_generations += 1
		if reward == 20:
			rep_letters += 1 
	print("Proportion of good generations: ", num_good_generations / len(generations))
	if check_rep:
		print("Proportion of desired generations (with desired letters): ", rep_letters / num_good_generations)


def get_parent_states(s):
	if not s:
		# parent states can be any of the 26 characters
		parent_states = [chr(i + ord[action_letters[0]]) for i in range(N)]
	else:
		parent_states = [s[:-1]]

	return parent_states


def train(gflownet, reward_func, num_epochs, num_letters_in_words, action_letters):
	optimizer = torch.optim.Adam(gflownet.parameters(), lr=3e-5)
	losses = []
	sampled_states = []
	grad_acc_step = 5  # Gradient accumulation steps
	trajectory_loss = 0
	N = len(action_letters)

	gflownet.train()
	for e in tqdm(range(num_epochs)):
		state = action_letters[random.randint(0, N-1)] #chr(random.randint(0, 25) + 97)
		flow = gflownet([state])
		for s in range(num_letters_in_words - 1):
			norm_flow = flow / flow.sum()
			next_action = Categorical(probs=norm_flow).sample().item()  # Sample action proportional to flow
			assert next_action >= 0 and next_action < N
			next_state = state + chr(next_action + ord(action_letters[0]))  # Convert int to char
			parent_states = get_parent_states(next_state)
			parent_flow = gflownet(parent_states)

			if s == num_letters_in_words - 2:
				# terminal state
				reward = reward_func.get_reward(next_state)
				flow = torch.zeros(N)  # No children
			else:
				reward = 0
				flow = gflownet(next_state)

			trajectory_loss += (parent_flow.sum() - flow.sum() - reward)**2
			state = next_state

		sampled_states.append(state)
		if e % grad_acc_step == 0:
			losses.append(trajectory_loss.item())
			optimizer.zero_grad()
			trajectory_loss.backward()
			optimizer.step()
			trajectory_loss = 0  # One epoch / episode --> one trajectory

	print("Training complete!")
	return sampled_states, losses


def plot_data(losses):
	plt.plot(losses)
	plt.yscale("log")
	plt.ylabel("Log loss")
	plt.xlabel("Training step")
	plt.title("Loss during training")
	plt.savefig("Loss.png")


def main():
	# Test hyperparams with RayTune
	reward_func = RepeatedLettersReward(extra_reward_letters=["a"])
	tkz = Tokenizer("google-bert/bert-base-uncased")  # Convert input strings to token IDs
	action_letters = ["a", "b", "c", "d", "e", "f"]  # Actions that the model can take -- permissible letters
	N = len(action_letters)

	# Instantiate model
	gflownet = GFlowNet(tkz, num_actions=N)


	num_epochs = 30000
	num_letters_in_words = 3  # Create three letter valid words
	sampled_states, losses = train(gflownet, reward_func, num_epochs, num_letters_in_words, action_letters)

	# Plot training loss curve
	plot_data(losses)

	# Show last few samples of words
	generations = sampled_states[-50:]
	print(generations)
	print("Random generation probability: ", math.factorial(N) / (math.factorial(N - num_letters_in_words) * (N ** N)))
	check_generations(generations, reward_func, check_rep=True)

if __name__ == "__main__":
	main()