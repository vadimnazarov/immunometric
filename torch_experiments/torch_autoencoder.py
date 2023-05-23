import random
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable 
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import glob
import string
import unicodedata

import numpy as np
import time
import math
import sys

all_letters = string.ascii_letters + " .,;'-@"
n_letters = len(all_letters)

N_EMBED = 32
N_HIDDEN = 64
N_LATENT = 40
SEQ_LEN = 20
N_EPOCHS = 300

BATCH_SIZE = 128

USE_CUDA = torch.cuda.is_available()


def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)


def vectorize(seqs):
	X = np.zeros((SEQ_LEN, len(seqs), n_letters), dtype=np.uint8)
	for i, sequence in enumerate(seqs):
		for t, letter in enumerate(sequence):
			X[t, i, all_letters.find(letter)] = 1
		for t in range(len(sequence), SEQ_LEN):
			X[t, i, all_letters.find("@")] = 1
	return X


def make_letter_indices(seqs):
	X = np.zeros((SEQ_LEN, len(seqs)), dtype=np.uint8)
	for i, sequence in enumerate(seqs):
		for t, letter in enumerate(sequence):
			X[t, i] = all_letters.find(letter)
		for t in range(len(sequence), SEQ_LEN):
			X[t, i] = all_letters.find("@")
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
	inds = np.random.permutation([x for x in range(len(seqs))])
	for i in range(len(seqs) - (len(seqs) % 16)):
		batches.append(sorted(inds))
	return batches


class Encoder(torch.nn.Module):

	def __init__(self, embed_dim = N_EMBED, gru_hidden_dim = N_HIDDEN, latent_dim = N_LATENT, drop=.3):
		super(Encoder, self).__init__()

		# self.embed = nn.Embedding(BATCH_SIZE, embed_dim)
		self.rnn = nn.GRU(n_letters, gru_hidden_dim, 2)
		self.linear = nn.Linear(gru_hidden_dim, latent_dim)

	def forward(self, input):
		# x = self.embed(input)
		# x = pack_padded_sequence(x, lengths)
		output, x = self.rnn(input, self.hidden)
		# x = F.elu(x.squeeze())
		x = F.elu(x[1].view((-1, N_HIDDEN)))
		x = self.linear(x)
		x = F.elu(x)
		return x

	def init_hidden(self, batch_size):
		hidden = autograd.Variable(torch.zeros(2, batch_size, N_HIDDEN))
		if USE_CUDA:
			self.hidden = hidden.cuda()
		else:
			self.hidden = hidden
		
	def init_weights(self):
		init.kaiming_normal(self.rnn.weight_ih_l0)
		init.kaiming_normal(self.rnn.weight_hh_l0)
		init.kaiming_normal(self.linear.weight)


class Decoder(torch.nn.Module):

	def __init__(self, latent_dim=N_LATENT, gru_hidden_dim=N_HIDDEN):
		super(Decoder, self).__init__()

		self.rnn = nn.GRU(latent_dim, gru_hidden_dim, 2)
		self.linear = nn.Linear(gru_hidden_dim, n_letters)


	def forward(self, input):
		output, _ = self.rnn(input, self.hidden)
		# print(output.size())
		x = torch.cat(output, 0)
		x = F.elu(x)
		x = self.linear(x)
		x = F.log_softmax(x)
		return x


	def init_hidden(self, batch_size):
		hidden = autograd.Variable(torch.zeros(2, batch_size, N_HIDDEN))
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
seq_arr = torch.from_numpy(vectorize([unicodeToAscii(x[0]) for x in temp])).type(torch.FloatTensor)
cat_vec = torch.from_numpy(make_letter_indices([unicodeToAscii(x[0]) for x in temp])).type(torch.LongTensor)
if USE_CUDA:
	seq_arr = seq_arr.cuda()
	cat_vec = cat_vec.cuda()

len_vec = np.array([len(x[0]) for x in temp])

encoder = Encoder()
decoder = Decoder()

if USE_CUDA:
	encoder = encoder.cuda()
	decoder = decoder.cuda()
encoder.init_weights()
decoder.init_weights()

criterion = nn.NLLLoss()

start = time.time()

loss_log = []
for iter in range(N_EPOCHS):
	print("Iteration:", iter+1)

	learning_rate = 1e-3 / (1 + iter // 60)
	print("lr:", learning_rate)
	optimizer_enc = torch.optim.RMSprop(encoder.parameters(), lr = learning_rate)
	optimizer_dec = torch.optim.RMSprop(decoder.parameters(), lr = learning_rate)

	all_loss = []

	ITER_PER_EPOCH = 20050 // BATCH_SIZE
	for ind in range(ITER_PER_EPOCH):
		optimizer_enc.zero_grad()
		optimizer_dec.zero_grad()

		inds = np.random.choice(20000, BATCH_SIZE, replace=False)
		inds.sort()
		# lengths = len_vec[inds]
		inds = torch.from_numpy(inds)
		if USE_CUDA:
			inds = inds.cuda()			
		# x, y = get_seq(inds)
		if USE_CUDA:
			x, y = (autograd.Variable(seq_arr.index_select(1, inds)).cuda(),
				autograd.Variable(cat_vec.index_select(1, inds)).view((-1,)).cuda())
		else:
			x, y = (autograd.Variable(seq_arr.index_select(1, inds)),
				autograd.Variable(cat_vec.index_select(1, inds)).view((-1,)))

		encoder.init_hidden(len(inds))
		encoder.zero_grad()
		y_latent = encoder(x)

		y_latent = y_latent.expand(SEQ_LEN, BATCH_SIZE, N_LATENT)
		decoder.init_hidden(len(inds))
		decoder.zero_grad()
		y_pred = decoder(y_latent)

		loss = criterion(y_pred, y)
		loss.backward()

		optimizer_enc.step()
		optimizer_dec.step()

		all_loss.append(loss.data[0])

		if ind == ITER_PER_EPOCH-1:
			ss_true = ["" for x in range(BATCH_SIZE)]
			ss_pred = ["" for x in range(BATCH_SIZE)]

			j = 0
			for i in range(y.size()[0]):
				ss_true[j] += all_letters[y[i].data[0]]
				_, index = y_pred[i].max(0)
				ss_pred[j] += all_letters[index.data[0]]
				j = (j + 1) % BATCH_SIZE
			for i, (s1, s2) in enumerate(zip(ss_true, ss_pred)):
				print(s1, "\n", s2, "\n", sep="")
				if i > 3:
					break


	tstr, sec = timeSince(start)
	print("sec/iter:", round(sec / (iter+1), 3))
	print("loss:", sum(all_loss))
	loss_log.append(sum(all_loss))

	print()

if len(sys.argv) == 1:
	sys.argv = sys.argv[0], "./"
with open(sys.argv[1] + "/log.txt", "w") as outf:
	for ll in loss_log:
		print(ll, file=outf)