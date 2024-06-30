from transformers import AutoTokenizer
import torch


class RepeatedLettersReward():
	def __init__(self, extra_reward_letters=None, base_reward=10, extra_reward=20):
		# Check if word has repeating letters
		# Give extra reward for some letters
		self.extra_reward_letters = extra_reward_letters
		self.base_reward = base_reward
		self.extra_reward = extra_reward

	def get_reward(self, generation):
		char_map = {}
		for c in generation:
			if c not in char_map:
				char_map[c] = True
			else:
				return 0  # Repeating char

		if self.extra_reward_letters is not None:
			for c in self.extra_reward_letters:
				if c in char_map:
					return self.extra_reward
		return self.base_reward


class Tokenizer():
	def __init__(self, tokenizer_model):
		self.tokenizer_model = AutoTokenizer.from_pretrained(tokenizer_model)

	def encode(self, word):
		return self.tokenizer_model.encode(word)[1]  # Remove start and end tokens
