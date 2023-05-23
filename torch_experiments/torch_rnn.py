print("START")


import random
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import glob
import string
import unicodedata

import numpy as np
import time
import math

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

N_EMBED = 32
N_HIDDEN = 32
N_LATENT = 64

BATCH_SIZE = 64

USE_CUDA = torch.cuda.is_available()


def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)


def vectorize(seqs):
	X = np.zeros((30, len(seqs), n_letters), dtype=np.uint8)
	for i, sequence in enumerate(seqs):
		for t, letter in enumerate(sequence):
			X[t, i, all_letters.find(letter)] = 1
	return X


def get_seq(ind):
	if USE_CUDA:
		return (autograd.Variable(seq_arr.index_select(1, ind)).cuda(),
				autograd.Variable(cat_vec[ind]).cuda())
	else:
		return (autograd.Variable(seq_arr.index_select(1, ind)),
				autograd.Variable(cat_vec[ind]))


def timeSince(since):
	now = time.time()
	sec = now - since
	s = sec
	m = math.floor(sec / 60)
	s -= m * 60
	return '%dm %ds' % (m, s), sec


def batchify(seqs, cats):
	batches = []
	inds = np.random.permutation([x for x in len(seqs)])
	for i in range(len(seqs) - (len(seqs) % 16)):
		batches.append(sorted(inds))
	return batches


class RNN(torch.nn.Module):

	def __init__(self, n_classes, gru_hidden_dim = N_HIDDEN):
		super(RNN, self).__init__()

		# self.embed = nn.Embedding(BATCH_SIZE, embed_dim)
		self.rnn = nn.GRU(n_letters, gru_hidden_dim)
		self.linear = nn.Linear(gru_hidden_dim, n_classes)

	def forward(self, input, lengths):
		# x = self.embed(input)
		# x = pack_padded_sequence(x, lengths)
		output, x = self.rnn(input, self.hidden)
		x = F.elu(x.squeeze())
		x = self.linear(x)
		x = F.log_softmax(x)
		return x

	def init_hidden(self, batch_size):
		hidden = autograd.Variable(torch.zeros(1, batch_size, N_HIDDEN))
		if USE_CUDA:
			self.hidden = hidden.cuda()
		else:
			self.hidden = hidden
		
	def init_weights(self):
		init.kaiming_normal(self.rnn.weight_ih_l0)
		init.kaiming_normal(self.rnn.weight_hh_l0)
		init.kaiming_normal(self.linear.weight)


file_paths = glob.glob("./data/names/*.txt")
len_vec = []
cat_vec = []
seq_vec = []
all_categories = []
cat_dict = {}
for file in file_paths:
	cat = file.split('/')[-1].split('.')[0]
	if cat not in cat_dict:
		cat_dict[cat] = len(cat_dict)
	with open(file, encoding='utf-8') as inp:
		for line in inp:
			seq_vec.append(line.strip())
			cat_vec.append(cat_dict[cat])
			all_categories.append(cat)
			len_vec.append(len(line.strip()))


temp = sorted(zip(seq_vec, cat_vec), reverse=True, key = lambda x: len(x[0]))
seq_arr = torch.from_numpy(vectorize([x[0] for x in temp])).type(torch.FloatTensor)
cat_vec = torch.from_numpy(np.array([x[1] for x in temp])).type(torch.LongTensor)
if USE_CUDA:
	seq_arr = seq_arr.cuda()
	cat_vec = cat_vec.cuda()

len_vec = np.array([len(x[0]) for x in temp])

rnn = RNN(len(cat_dict))

if USE_CUDA:
	rnn = rnn.cuda()
rnn.init_weights()

criterion = nn.NLLLoss()

start = time.time()

for iter in range(150):
	print("Iteration:", iter+1)

	learning_rate = 1e-2 / (1 + iter // 50)
	print("lr:", learning_rate)
	optimizer = torch.optim.RMSprop(rnn.parameters(), lr = learning_rate)

	all_loss = []

	for ind in range(128):
		optimizer.zero_grad()

		inds = np.random.choice(20000, BATCH_SIZE, replace=False)
		inds.sort()
		lengths = len_vec[inds]
		inds = torch.from_numpy(inds)
		if USE_CUDA:
			inds = torch.from_numpy(inds).cuda()			
		x, y = get_seq(inds)

		rnn.init_hidden(len(inds))
		rnn.zero_grad()
		y_pred = rnn(x, lengths)
		loss = criterion(y_pred, y)
		loss.backward()

		optimizer.step()

		all_loss.append(loss.data[0])

	tstr, sec = timeSince(start)
	print("sec/iter:", round(sec / (iter+1), 3))
	print("loss:", sum(all_loss))

	print()