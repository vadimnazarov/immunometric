from argparse import ArgumentParser
import time
import sys
import os

import pandas as pd
import numpy as np
from numpy.random import randint

import torch
from torch import from_numpy, save, load, cuda, nn
from torch import sort as torch_sort
from torch.optim import RMSprop
from torch.nn.init import kaiming_uniform
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader

from immunnet.utils import *
from immunnet.model import Autoencoder, StackedAutoencoder, VariationalAutoencoder, StackedVAE



class SelectionModel(nn.Module):

    def __init__(self, input_dim, rnn_hid_dim, n_layers, dropout):
        super(SelectionModel, self).__init__()

        self._hidden_dim = rnn_hid_dim
        self._n_layers = n_layers
        self._dropout_rnn = dropout[0]
        self._dropout_lin = dropout[1]

        self.rnn = nn.GRU(input_dim, self._hidden_dim, n_layers, bidirectional=True, dropout=self._dropout_rnn)
        # self.linear1 = nn.Linear(self._hidden_dim, self._hidden_dim)
        # self.dropout1 = nn.Dropout(self._dropout_lin)
        self.final = nn.Linear(self._hidden_dim, 1)


    def forward(self, input):
        _, x = self.rnn(input)
        x = (x[-1] + x[-2]).mul_(.5).view((-1, self._hidden_dim))
        x = F.elu(x)

        # x = self.linear1(x)
        # x = F.elu(x)
        # x = self.dropout1(x)

        x = self.final(x)
        return F.sigmoid(x)


    def init_weights(self):
        kaiming_uniform(self.rnn.weight_ih_l0)
        kaiming_uniform(self.rnn.weight_hh_l0)
        # kaiming_uniform(self.linear1.weight)
        kaiming_uniform(self.final.weight)


    def criterion():
        def _criterion(y_pred, y_true):
            return (F.binary_cross_entropy(y_pred, y_true), )

        return _criterion


if __name__ == "__main__":
    alphabet = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '$']
    char_indices = dict((c, i) for i, c in enumerate(alphabet))
    indices_char = dict((i, c) for i, c in enumerate(alphabet))

    INPUT_DIM = len(alphabet)


    parser = ArgumentParser()

    parser.add_argument("--input_exp_train", 
        type=str, default="A1_train.txt.zip", 
        help="Experimental input sequences")
    parser.add_argument("--input_exp_test", 
        type=str, default="A1_test.txt.zip", 
        help="Experimental input sequences")

    parser.add_argument("--input_gen_train", 
        type=str, default="Gen_train.txt.zip", 
        help="Experimental input sequences")
    parser.add_argument("--input_gen_test", 
        type=str, default="Gen_test.txt.zip", 
        help="Experimental input sequences")

    parser.add_argument("--output", "-o", 
        type=str, default="./results/", 
        help="Output folder")
    parser.add_argument("--load", "-l", 
        type=str, default="", 
        help="Path to the *.pt filename with model's weights")
    # parser.add_argument("--model", "-m", 
    #     type=str, default="ae", 
    #     help="Models: ae (simple autoencoder), sae (stacked autoencoder), vae (variational autoencoder)")
    # parser.add_argument("--train_size", "--train", 
    #     type=int, default=1500000, 
    #     help="Number of training examples")
    parser.add_argument("--megabatch_size", "--mb", 
        type=int, default=100000, 
        help="Number of training examples per epoch (to fit into GPU memory")
    parser.add_argument("--epochs_per_megabatch", "--epoch_megabatch", 
        type=int, default=20, 
        help="Number of epochs per mega-batch of 'megabatch_size' size")

    # parser.add_argument("--test_size", "--test", 
    #     type=int, default=40000, 
    #     help="Number of testing examples (overall)")
    # parser.add_argument("--cv_size", "--cv", 
    #     type=int, default=40000, 
    #     help="Number of cross-validation examples (overall)")

    parser.add_argument("--batch_size", "-b", 
        type=int, default=64, 
        help="Size of the batch")
    parser.add_argument("--epochs", "-e", "--epoch",
        type=int, default=100, 
        help="Number of epochs")
    parser.add_argument("--learning_rate", "--lr", 
        type=float, default=1e-2, 
        help="Starting learning rate")
    parser.add_argument("--prefix", "--pr", 
        type=str, default="", 
        help="Prefix to folder")

    args = parser.parse_args()


    def read_seq_file(filename):
        seq_vec = pd.read_table(filename)["cdr3aa"] + '$'
        len_vec = seq_vec.apply(len).values

        sorted_inds = sorted(enumerate(len_vec), key=lambda x: x[1], reverse=True)
        seq_vec = seq_vec[[x[0] for x in sorted_inds]]
        len_vec = len_vec[[x[0] for x in sorted_inds]]

        return seq_vec, len_vec


    print("Loading the sequence data...", end="\t")
    start = time.time()
    seq_exp_X_train, len_exp_X_train = read_seq_file(args.input_exp_train)
    seq_exp_X_test,  len_exp_X_test =  read_seq_file(args.input_exp_test)
    seq_gen_X_train, len_gen_X_train = read_seq_file(args.input_gen_train)
    seq_gen_X_test,  len_gen_X_test =  read_seq_file(args.input_gen_test)

    print(np.intersect1d(seq_exp_X_train, seq_exp_X_test))
    print(np.intersect1d(seq_gen_X_train, seq_gen_X_test))
    print(np.intersect1d(seq_exp_X_train, seq_gen_X_train))
    print(np.intersect1d(seq_exp_X_test, seq_gen_X_test))

    print("Vectorizing data...", end="\t")
    start = time.time()

    do_megabatches = True
    if args.megabatch_size >= len(seq_gen_X_train):
        args.megabatch_size = len(seq_gen_X_train)
        do_megabatches = False

    train_indices = np.random.choice(len(seq_gen_X_train), args.megabatch_size, replace=False)
    X_exp_train, _, _ = vectorize_torch(seq_exp_X_train, len_exp_X_train, alphabet, char_indices, np.arange(0, len(seq_exp_X_train)))
    X_exp_test,  _, _  = vectorize_torch(seq_exp_X_test,  len_exp_X_test,  alphabet, char_indices, np.arange(0, len(seq_exp_X_test)))
    X_gen_train, _, len_gen_X_train_subset = vectorize_torch(seq_gen_X_train, len_gen_X_train, alphabet, char_indices, train_indices)
    X_gen_test,  _, _  = vectorize_torch(seq_gen_X_test,  len_gen_X_test,  alphabet, char_indices, np.arange(0, len(seq_gen_X_test)))

    print(X_exp_train.size())
    print(X_exp_test.size())
    print(X_gen_train.size())
    print(X_gen_test.size())

    model_dir_name = ["rnn_sel"]

    if args.load:
        print("Loading the model from", args.load)
        model = load(args.load)
    else:
        model = SelectionModel(len(alphabet), 64, 2, (.3, .3))
        model.init_weights()

    if USE_CUDA:
        model.cuda()
        

    criterion = SelectionModel.criterion()

    loss_log = {"loss_train": [], "loss_test": [], "acc_train": [], "acc_test": []}
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

    print(torch_summarize(model))

    print("Training...")
    training_start = time.time()
    for i_epoch in range(args.epochs):
        print("Epoch:", i_epoch+1, "/", args.epochs)

        if do_megabatches and ((i_epoch + 1) % args.epochs_per_megabatch == 0):
            print("Re-vectorizing data...", end="\t")
            vector_start = time.time()
            train_indices_gen = np.random.choice(len(seq_gen_X_train), args.megabatch_size, replace=False)
            X_gen_train, _, len_gen_X_train_subset = vectorize_torch(seq_gen_X_train, len_gen_X_train, alphabet, char_indices, train_indices)
            print("Done in", time_since(vector_start)[0] + ".")

        learning_rate = args.learning_rate / (3 ** (i_epoch // 10))
        print("lr:", learning_rate)
        optimizer = RMSprop(model.parameters(), lr = learning_rate)

        cur_loss = {x:[] for x in loss_log}
        correct, total = 0, 0

        y_true = to_var(torch.FloatTensor([1 for _ in range(args.batch_size)] + [0 for _ in range(args.batch_size)]).view((-1, 1)))

        for i_batch, (x1, x2) in enumerate(batchify_selection(X_exp_train, X_gen_train, len_exp_X_train, len_gen_X_train_subset, args.batch_size)):
            model.zero_grad()
            optimizer.zero_grad()

            y_pred1 = model(x1)
            y_pred2 = model(x2)

            y_pred = torch.cat([y_pred1, y_pred2], 0)

            loss = criterion(y_pred, y_true)
            loss[0].backward()
            optimizer.step()

            train_loss_dict = wrap_loss(loss, WRAP_TRAIN)
            for key, val in train_loss_dict.items():
                cur_loss[key].append(val)

            # Compute accuracy
            y = y_pred.data
            y[y >= .5] = 1
            y[y < .5] = 0
            
            y2 = y_true.data.long()
            total += y2.size(0)
            correct += (y.long() == y2).sum()
            # print((y.long() == y2).sum() / y2.size(0))

        print("train", correct, total)
        cur_loss["acc_train"].append(100 * correct / total)

        # Compute the test loss
        test_loss_dict = test_loss_selection(model, criterion, X_exp_test, len_exp_X_test, X_gen_test, len_gen_X_test)
        for key, val in test_loss_dict.items():
            cur_loss[key].append(val)

        tstr, sec = time_since(training_start)
        print("sec/epoch:", round(sec / (i_epoch+1), 3))
        print("sec/batch:", round(sec / (i_epoch+1) / (i_batch+1), 3), "(" + str(i_batch+1), "batches)")
        for loss_key in sorted(cur_loss.keys()):
            loss_log[loss_key].append(sum(cur_loss[loss_key]))
            print(loss_key, ": ", sum(cur_loss[loss_key]), sep="")

            filename = loss_files[loss_key]
            with open(filename, "a") as outf:
                outf.write(str(sum(cur_loss[loss_key])) + "\n")
        print()

        # Save latent representation and model weights 
        if ((i_epoch+1) % 50) == 0:
            print("Saving the model...", end="\t")
            save_start = time.time()
            save(model, model_dir_name + "/model." +  str(i_epoch+1) + ".pt")
            print("Done in", time_since(save_start)[0] + ".\n")

    # Save final latent representation and model weights.
    print("Saving the final model...", end="\t")
    save_start = time.time()
    save(model, model_dir_name + "/model.final.pt")
    print("Done in", time_since(save_start)[0] + ".\n")