from argparse import ArgumentParser
import time
import sys
import os
import gzip

import pandas as pd
import numpy as np
from numpy.random import randint

from torch import from_numpy, save, load, cuda
from torch import sort as torch_sort
from torch.optim import RMSprop, Adam
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# from torch.utils.data import TensorDataset, DataLoader

from immunnet.utils import *
from immunnet.model import Autoencoder, StackedAutoencoder, VariationalAutoencoder, StackedVAE


import torch
from torch import randn, load, Tensor
from torch import sum as torch_sum
from torch.nn import Module, Sequential, Linear, GRU, NLLLoss, BatchNorm1d, Dropout
from torch.nn.init import kaiming_uniform
import torch.nn.functional as F


class WordsVAE(Module):

    def __init__(self, input_dim, latent_dim, rnn_hid_dim, dropout):
        super(WordsVAE, self).__init__()

        self._latent_dim = latent_dim
        self._hidden_dim = rnn_hid_dim
        self._dropout = dropout

        self.e_linear_in = Linear(input_dim, self._hidden_dim)
        self.e_drop_in = Dropout(dropout)

        self.e_linear1 = Linear(self._hidden_dim, self._hidden_dim)
        self.e_drop1 = Dropout(dropout)

        # self.e_linear2 = Linear(self._hidden_dim, self._hidden_dim)
        # self.e_drop2 = Dropout(dropout)

        self.e_final = Linear(self._hidden_dim, self._latent_dim)


        self.d_linear_in = Linear(self._latent_dim, self._hidden_dim)
        self.d_drop_in = Dropout(dropout)

        self.d_linear1 = Linear(self._hidden_dim, self._hidden_dim)
        self.d_drop1 = Dropout(dropout)

        self.d_final = Linear(self._hidden_dim, input_dim)


    def forward(self, input):
        x = self.encode(input)
        output = self.decode(x)
        return output


    def encode(self, input):
        x= self.e_linear_in(input)
        x = F.elu(x)
        x = self.e_drop_in(x)

        x = self.e_linear1(x)
        x = F.elu(x)
        x = self.e_drop1(x)

        x = self.e_final(x)
        return x


    def decode(self, input):
        x = self.d_linear_in(input)
        x = F.elu(x)
        x = self.d_drop_in(x)

        x = self.d_linear1(x)
        x = F.elu(x)
        x = self.d_drop1(x)

        x = self.d_final(x)
        return x


    def init_weights(self):
        kaiming_uniform(self.e_linear_in.weight)
        kaiming_uniform(self.e_linear1.weight)
        # kaiming_uniform(self.e_linear2.weight)
        kaiming_uniform(self.e_final.weight)

        kaiming_uniform(self.d_linear_in.weight)
        kaiming_uniform(self.d_linear1.weight)
        # kaiming_uniform(self.d_linear2.weight)
        kaiming_uniform(self.d_final.weight)
        


def criterion_fun(y_pred, y_true):
    rec_loss = F.mse_loss(y_pred, y_true, size_average=True)
    return (rec_loss, )


def batchify_words(X, batch_size):
    inds = from_numpy(np.random.permutation(X.size(0)))

    if USE_CUDA:
        inds = inds.cuda()

    for i in range(X.size(0) // batch_size):
        batch_inds = inds[i*batch_size : (i+1)*batch_size]
        yield X[batch_inds]


def save_embed_words(filename, model, X, seq, batch_size):
    model.eval()
    with gzip.open(filename+".gz", "wt") as outf:
        for batch_inds in batchify_straight(X.size(0), batch_size):
            # batch_inds = batch_inds.cpu()
            y_pred = model.encode(to_var(X[batch_inds]))
            # y_pred = model.encode_(to_var(X[batch_inds]))

            if type(y_pred) is tuple: 
                y_pred = y_pred[0]

            for i_seq in range(len(batch_inds)):
                if seq is not None:
                    # print(seq[batch_inds[i_seq]]["noun"], " ".join(map(lambda z: str(round(z, 5)), y_pred[i_seq].data)), file = outf)
                    print(seq[batch_inds[i_seq]], " ".join(map(lambda z: str(round(z, 5)), y_pred[i_seq].data)), file = outf)
                else:
                    print("NONAME", " ".join(map(lambda z: str(round(z, 5)), y_pred[i_seq].data)), file = outf)
    model.train()


if __name__ == "__main__":
    alphabet = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '$']
    char_indices = dict((c, i) for i, c in enumerate(alphabet))
    indices_char = dict((i, c) for i, c in enumerate(alphabet))


    parser = ArgumentParser() #7 epoch - 0.0021 #7 epoch - 0.008

    parser.add_argument("--input", "-i", 
        type=str, help="Input data")
    parser.add_argument("--output", "-o", 
        type=str, default="./results/", 
        help="Output folder")
    parser.add_argument("--load", "-l", 
        type=str, default="", 
        help="Path to the *.pt filename with model's weights")
    parser.add_argument("--batch_size", "-b", 
        type=int, default=256, 
        help="Size of the batch")
    parser.add_argument("--epochs", "-e", "--epoch",
        type=int, default=20, 
        help="Number of epochs")
    parser.add_argument("--latent_dim", "--ld", "--latent", "--lat", 
        type=int, default=30, 
        help="Number of latent dimensions")
    parser.add_argument("--hidden_dim", "--hd", "--hidden", 
        type=int, default=32, 
        help="GRU hidden layer size")
    parser.add_argument("--layers", "--lay", 
        type=int, default=2, 
        help="Number of layers for GRU")
    parser.add_argument("--dropout", "--drop", 
        type=str, default=".35,.3", 
        help="Dropout for GRU (single value) and linear layers in case of SAE / VAE (comma-separated values)")
    parser.add_argument("--learning_rate", "--lr", 
        type=float, default=3e-3, 
        help="Starting learning rate")
    parser.add_argument("--prefix", "--pr", 
        type=str, default="", 
        help="Prefix to folder")

    args = parser.parse_args()


    ##################
    ###  Training  ###
    ##################
    np.random.seed(42)

    print("Loading the data...")
    words_file = args.input
    X_train = np.load(words_file)["matrix"]
    print(X_train.shape)

    nonz = []
    for i in range(X_train.shape[1]):
        if X_train[:,i].sum() != 0:
            nonz.append(i)
    X_train = X_train[:, nonz]
    print("Left", len(nonz), "nonzero dimensions")
    print(X_train.shape)

    X_train = from_numpy(X_train).contiguous()
    
    train_names = np.load(words_file)["names"]

    print(" -- X train shape:", X_train.size())

    model_dir_name = ["ae_words"]

    model_dir_name.append("lat" + str(args.latent_dim))
    model_dir_name.append("hid" + str(args.hidden_dim))
    if args.prefix:
        model_dir_name.append(args.prefix)

    if args.load:
        print("Loading the model from", args.load)
        model = load(args.load)
    else:
        model = WordsVAE(X_train.size(1), 
            args.latent_dim, 
            args.hidden_dim, 
            tuple(map(float, str(args.dropout).split(",")))[0])
        model.init_weights()

    if USE_CUDA:
        model.cuda()
        X_train = X_train.cuda()

    criterion = criterion_fun
    model_dir_name[0] += "_lay2" 

    loss_log = {"loss_train": []}
    loss_files = {}


    model_dir_name = (args.output + "/" + "_".join(map(str, model_dir_name)) + "/").replace("//", "/")
    if not os.path.exists(model_dir_name):
        print("Creating new input directory ", model_dir_name, end = "\t")
        os.makedirs(model_dir_name)
    else:
        print("Rewriting the input directory", model_dir_name, end = "\t")
    print("OK.")
    print()

    for loss_key in loss_log:
        filename = model_dir_name + "/log." + loss_key + ".txt"
        loss_files[loss_key] = filename
        with open(filename, "w") as _:
            pass

    with open(model_dir_name + "args.txt", "w") as inp:
        for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
            print(key, val, file=inp)

    print("Training...")
    for i_epoch in range(args.epochs):
        training_start = time.time()

        print("Epoch:", i_epoch+1, "/", args.epochs, "[", model_dir_name, "]")

        learning_rate = args.learning_rate / (10 ** (i_epoch // 10))
        print("lr:", learning_rate)
        optimizer = Adam(model.parameters(), lr = learning_rate)

        cur_loss = {x:[] for x in loss_log}

        for i_batch, x in enumerate(batchify_words(X_train, args.batch_size)):
            model.zero_grad()
            optimizer.zero_grad()

            x = to_var(x)

            y_pred = model(x)
            loss = criterion(y_pred, x)

            loss[0].backward()
            optimizer.step()

            train_loss_dict = wrap_loss(loss, WRAP_TRAIN)
            for key, val in train_loss_dict.items():
                cur_loss[key].append(val)

        tstr, sec = time_since(training_start)

        print("sec/epoch:", round(sec / (i_epoch+1), 3), "(sec/batch:", round(sec / (i_epoch+1) / (i_batch+1), 3), "[" + str(i_batch+1), "batches])")
        for loss_key in sorted(cur_loss.keys()):
            loss_log[loss_key].append(sum(cur_loss[loss_key]))
            print(loss_key, ": ", sum(cur_loss[loss_key]), sep="")

            filename = loss_files[loss_key]
            with open(filename, "a") as outf:
                outf.write(str(sum(cur_loss[loss_key])) + "\n")
        print()

    # Save final latent representation and model weights.
    print("Writing final latent vectors and saving the final model...", end="\t")
    save_start = time.time()
    save_embed_words(model_dir_name + "/embeddings_train.final.txt", model, X_train, train_names, 1024)
    # save_embed_words(model_dir_name + "/embeddings_test.final.txt", model, X_test.X, test_names, 1024)

    # save(model, model_dir_name + "/model.final.pt")
    print("Done in", time_since(save_start)[0] + ".\n")