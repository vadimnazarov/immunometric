from argparse import ArgumentParser
import time
import sys
import os

import pandas as pd
import numpy as np
from numpy.random import randint

from torch import from_numpy, save, load, cuda
from torch import sort as torch_sort
from torch.optim import RMSprop
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# from torch.utils.data import TensorDataset, DataLoader

from immunnet.utils import *
from immunnet.model import Autoencoder, StackedAutoencoder, VariationalAutoencoder, StackedVAE


import torch
from torch import randn, load, Tensor
from torch import sum as torch_sum
from torch.nn import Module, ModuleList, Linear, GRU, NLLLoss, BatchNorm1d, Dropout
from torch.nn.init import kaiming_uniform, kaiming_uniform
import torch.nn.functional as F



class TestDatasetSiamese():

    def __init__(self, X, seqs, lens, batch_size):
        self.X = X
        self.seqs = seqs
        self.lens = lens
        self.bs = batch_size
        self.inds_bal = []
        self.inds_dis = []

        for i in range(self.X.size(1) // self.bs + 1):
            inds_center = np.sort(np.random.choice(self.X.size(1), size=self.bs // 4, replace=False))
            inds_left =   np.sort(np.random.choice(self.X.size(1), size=self.bs * 3 // 4, replace=True))
            inds_right =  np.sort(np.random.choice(self.X.size(1), size=self.bs * 3 // 4, replace=True))

            if USE_CUDA:
                self.inds_bal.append((inds_center, inds_left, inds_right, 
                from_numpy(inds_center).cuda(), from_numpy(inds_left).cuda(), from_numpy(inds_right).cuda()))
            else:
                self.inds_bal.append((inds_center, inds_left, inds_right, 
                from_numpy(inds_center), from_numpy(inds_left), from_numpy(inds_right)))


        for i in range(self.X.size(1) // self.bs + 1):
            inds_left =   np.sort(np.random.choice(self.X.size(1), size=self.bs, replace=True))
            inds_right =  np.sort(np.random.choice(self.X.size(1), size=self.bs, replace=True))

            if USE_CUDA:
                self.inds_dis.append((inds_left, inds_right, from_numpy(inds_left).cuda(), from_numpy(inds_right).cuda()))
            else:
                self.inds_dis.append((inds_left, inds_right, from_numpy(inds_left), from_numpy(inds_right)))


    def __len__(self):
        return len(self.inds_bal) + len(self.inds_dis)


    def get_batch(self):
        for inds_center, inds_left, inds_right, inds_center_cuda, inds_left_cuda, inds_right_cuda in self.inds_bal:
            yield self._get_bal(inds_center, inds_left, inds_right, inds_center_cuda, inds_left_cuda, inds_right_cuda)
        for inds_left, inds_right, inds_left_cuda, inds_right_cuda in self.inds_dis:
            yield self._get_dis(inds_left, inds_right, inds_left_cuda, inds_right_cuda)


    def _get_bal(self, inds_center, inds_left, inds_right, inds_center_cuda, inds_left_cuda, inds_right_cuda):
        # inds_center_cuda = from_numpy(inds_center)
        # inds_left_cuda = from_numpy(inds_left)
        # inds_right_cuda = from_numpy(inds_right)
        # if USE_CUDA:
        #     inds_center_cuda = inds_center_cuda.cuda()
        #     inds_left_cuda = inds_left_cuda.cuda()
        #     inds_right_cuda = inds_right_cuda.cuda()

        return ((pack_padded_sequence(to_var(self.X.index_select(1, inds_left_cuda)), 
                                    self.lens[inds_left]), 
               pack_padded_sequence(to_var(self.X.index_select(1, inds_right_cuda)), 
                                    self.lens[inds_right]), 
               to_var(Tensor([edit_distance(x,y) for x,y in zip(self.seqs[inds_left], self.seqs[inds_right])]))), 
               (pack_padded_sequence(to_var(self.X.index_select(1, inds_center_cuda)), 
                                    self.lens[inds_center]), 
               to_var(Tensor([0 for _ in self.seqs[inds_center]]))))


    def _get_dis(self, inds_left, inds_right, inds_left_cuda, inds_right_cuda):
        # inds_left_cuda = from_numpy(inds_left)
        # inds_right_cuda = from_numpy(inds_right)
        # if USE_CUDA:
        #     inds_left_cuda = inds_left_cuda.cuda()
        #     inds_right_cuda = inds_right_cuda.cuda()

        return (pack_padded_sequence(to_var(self.X.index_select(1, inds_left_cuda)), 
                                    self.lens[inds_left]), 
               pack_padded_sequence(to_var(self.X.index_select(1, inds_right_cuda)), 
                                    self.lens[inds_right]), 
               to_var(Tensor([edit_distance(x,y) for x,y in zip(self.seqs[inds_left], self.seqs[inds_right])])))          


class SiameseNetwork(Module):

    def __init__(self, model, regression_dim, regression_layers, dropout):
        super(SiameseNetwork, self).__init__()

        self._model = model
        # self._reg_dim = regression_dim

        # self.reg_layers = ModuleList()
        # prev_dim = self._model._latent_dim * 2
        # for i in range(regression_layers):
        #     self.reg_layers.append(Linear(prev_dim, regression_dim))
        #     prev_dim = regression_dim
        # self.reg_layers.append(Linear(prev_dim, 1))  # final layer

        # self._dropout = dropout
        # self.drop_layers = ModuleList()
        # for i in range(len(self.reg_layers)):
        #     self.drop_layers.append(Dropout(self._dropout))


    # Predict for autoencoders
    def forward(self, in_left, in_right):
        x1 = self.encode(in_left)
        x2 = self.encode(in_right)
            
        return x1, x2


    # Predict for semiVAE
    # def forward(self, in_left, in_right):
    #     mu1, log_var1 = self.encode(in_left)
    #     mu2, log_var2 = self.encode(in_right)
            
    #     return mu1, log_var1, mu2, log_var2


    def encode(self, input):
        return self._model.encode(input)


    def encode_(self, input):
        return self._model.encode_(input)


    def init_weights(self):
        self._model.init_weights()

        # for i in range(len(self.reg_layers)):
        #     kaiming_uniform(self.reg_layers[i].weight)


    def criterion(loss_type, target_type):
        if loss_type == "mse":
            loss_fun = F.mse_loss
        elif loss_type == "l1":
            loss_fun = F.l1_loss
        else:
            print("ERROR: unknown loss function")

        if target_type == "loglev":
            target_fun = lambda z1, z2: (z1 - z2).abs().sum(1).exp()
        elif target_type == "lev":
            target_fun = lambda z1, z2: (z1 - z2).abs().sum(1)
        else:
            print("ERROR: unknown target function")

        # Regression layers
        # def _criterion_vae_very_old(y_pred, y_true):
        #     y_pred, mu1, log_var1, mu2, log_var2 = y_pred

        #     mu = torch.cat([mu1, mu2], 0)
        #     log_var = torch.cat([log_var1, log_var2], 0)

        #     kl_divergence = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        #     kl_divergence = torch_sum(kl_divergence).mul_(-.5)

        #     rec_loss = loss_fun(y_pred.view((-1,)), y_true, size_average=False)

        #     total_loss = rec_loss + kl_divergence
        #     return total_loss, rec_loss, kl_divergence


        # Prediction of log L
        def _criterion_vae(y_pred, y_true):
            mu1, log_var1, mu2, log_var2 = y_pred
            mu = torch.cat([mu1, mu2], 0)
            log_var = torch.cat([log_var1, log_var2], 0)

            kl_divergence = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            kl_divergence = torch_sum(kl_divergence).mul_(-.5)

            y_pred = target_fun(mu1, mu2)
            rec_loss = loss_fun(y_pred.view((-1,)), y_true, size_average=False)

            total_loss = rec_loss + kl_divergence
            return total_loss, rec_loss, kl_divergence


        def _criterion_novae(y_pred, y_true):
            mu1, mu2 = y_pred

            y_pred = target_fun(mu1, mu2)
            rec_loss = loss_fun(y_pred.view((-1,)), y_true, size_average=False)

            return (rec_loss, )

        return _criterion_novae



    def criterion_rep(self, loss_type, target_type):
        reparametrize = self._model.reparametrize

        if loss_type == "mse":
            loss_fun = F.mse_loss
        elif loss_type == "l1":
            loss_fun = F.l1_loss
        else:
            print("ERROR: unknown loss function")

        if target_type == "loglev":
            target_fun = lambda z1, z2: (z1 - z2).abs().sum(1).exp()
        elif target_type == "lev":
            target_fun = lambda z1, z2: (z1 - z2).abs().sum(1)
        else:
            print("ERROR: unknown target function")

        def _criterion_vae_new(y_pred, y_true):
            mu1, log_var1, mu2, log_var2 = y_pred
            mu = torch.cat([mu1, mu2], 0)
            log_var = torch.cat([log_var1, log_var2], 0)

            kl_divergence = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            kl_divergence = torch_sum(kl_divergence).mul_(-.5)

            z1 = reparametrize(mu1, log_var1)
            z2 = reparametrize(mu2, log_var2)

            y_pred = target_fun(z1, z2)
            rec_loss = loss_fun(y_pred.view((-1,)), y_true, size_average=False)

            total_loss = rec_loss + kl_divergence
            return total_loss, rec_loss, kl_divergence

        return _criterion_vae_new


    # Regression layers
    # def mean_error(y_pred, y_true):
    #     y_pred, _, _, _, _ = y_pred
    #     rec_loss = F.l1_loss(y_pred.view((-1,)), y_true, size_average=True)
    #     return rec_loss


    # Prediction of log L
    def mean_error_norep(loss_type, target_type):
        if loss_type == "mse":
            loss_fun = F.mse_loss
        elif loss_type == "l1":
            loss_fun = F.l1_loss
        else:
            print("ERROR: unknown loss function")

        if target_type == "loglev":
            target_fun = lambda z1, z2: (z1 - z2).abs().sum(1).exp()
        elif target_type == "lev":
            target_fun = lambda z1, z2: (z1 - z2).abs().sum(1)
        else:
            print("ERROR: unknown target function")

        def mean_error_novae(y_pred, y_true):
            mu1, mu2 = y_pred
            y_pred = target_fun(mu1, mu2)
            rec_loss = loss_fun(y_pred.view((-1,)), y_true, size_average=True)
            return rec_loss

        def mean_error_vae(y_pred, y_true):
            mu1, log_var1, mu2, log_var2 = y_pred
            y_pred = target_fun(mu1, mu2)
            rec_loss = loss_fun(y_pred.view((-1,)), y_true, size_average=True)
            return rec_loss

        return mean_error_novae


    def mean_error_rep(self, loss_type, target_type):
        reparametrize = self._model.reparametrize

        if loss_type == "mse":
            loss_fun = F.mse_loss
        elif loss_type == "l1":
            loss_fun = F.l1_loss
        else:
            print("ERROR: unknown loss function")

        if target_type == "loglev":
            target_fun = lambda z1, z2: (z1 - z2).abs().sum(1).exp()
        elif target_type == "lev":
            target_fun = lambda z1, z2: (z1 - z2).abs().sum(1)
        else:
            print("ERROR: unknown target function")

        def _mean_error(y_pred, y_true):
            mu1, log_var1, mu2, log_var2 = y_pred
            z1 = reparametrize(mu1, log_var1)
            z2 = reparametrize(mu2, log_var2)
            y_pred = target_fun(z1, z2)
            rec_loss = loss_fun(y_pred.view((-1,)), y_true, size_average=True)
            return rec_loss

        return _mean_error


    def description(self):
        orig = self._model.description()
        description = """Siamese Network\n"""
        description += "Model: " + orig
        return description


if __name__ == "__main__":
    alphabet = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '$']
    char_indices = dict((c, i) for i, c in enumerate(alphabet))
    indices_char = dict((i, c) for i, c in enumerate(alphabet))

    INPUT_DIM = len(alphabet)


    parser = ArgumentParser()

    parser.add_argument("--input_train", "-i1", 
        type=str, default="disser_data/Gen_train.txt", 
        help="Input data for training")
    parser.add_argument("--input_test", "-i2", 
        type=str, default="disser_data/Gen_test.txt", 
        help="Input data for testing")
    parser.add_argument("--output", "-o", 
        type=str, default="./results/", 
        help="Output folder")
    parser.add_argument("--load", "-l", 
        type=str, default="", 
        help="Path to the *.pt filename with model's weights")
    parser.add_argument("--model", "-m", 
        type=str, default="ae", 
        help="Models: ae (simple autoencoder), sae (stacked autoencoder), vae (variational autoencoder)")
    parser.add_argument("--megabatch_size", "--mb", 
        type=int, default=150000, 
        help="Number of training examples per epoch (to fit into GPU memory")
    parser.add_argument("--epochs_per_megabatch", "--epoch_megabatch", 
        type=int, default=20, 
        help="Number of epochs per mega-batch of 'megabatch_size' size")
    parser.add_argument("--batch_size", "-b", 
        type=int, default=256, 
        help="Size of the batch")
    parser.add_argument("--epochs", "-e", "--epoch",
        type=int, default=60, 
        help="Number of epochs")
    parser.add_argument("--loss_type",
        type=str, default="l1", 
        help="Type of the loss. 'l1' for L1, 'mse' for MSE.")
    parser.add_argument("--target_type",
        type=str, default="loglev", 
        help="Type of the target. 'loglev' for LogLevenshtein, 'lev' for Levenshtein.")
    parser.add_argument("--latent_dim", "--ld", 
        type=int, default=30, 
        help="Number of latent dimensions")
    parser.add_argument("--hidden_dim", "--hd", 
        type=int, default=32, 
        help="GRU hidden layer size")
    parser.add_argument("--layers", "--lay", 
        type=int, default=2, 
        help="Number of layers for GRU")
    parser.add_argument("--regression_dim", "--rd", 
        type=int, default=64, 
        help="Regression layers' size")
    parser.add_argument("--regression_layers", "--rl", 
        type=int, default=0, 
        help="Number of regression layers")
    parser.add_argument("--dropout", "--drop", 
        type=str, default=".35,.3", 
        help="Dropout for GRU (single value) and linear layers in case of SAE / VAE (comma-separated values)")
    parser.add_argument("--learning_rate", "--lr", 
        type=float, default=1e-3, 
        help="Starting learning rate")
    parser.add_argument("--balancing_mode", "--balance", 
        type=int, default=0, 
        help="1 for balanced batches, 0 for disbalanced")
    parser.add_argument("--prefix", "--pr", 
        type=str, default="", 
        help="Prefix to folder")
    parser.add_argument("--inference", "--inf", 
        type=str, default="", 
        help="Path to the input file with sequences. Parameter --load must be defined.")

    args = parser.parse_args()

    ###################
    ###  Inference  ###
    ###################
    if args.inference:
        start = time.time()

        print("Loading the sequence data...")
        seq_vec = pd.read_table(args.inference, sep=",")["cdr3aa"] + '$'
        len_vec = seq_vec.apply(len).values

        sorted_inds = sorted(enumerate(len_vec), key=lambda x: x[1], reverse=True)
        seq_vec = seq_vec[[x[0] for x in sorted_inds]]
        len_vec = len_vec[[x[0] for x in sorted_inds]]

        model_dir_name = ["siam", "inference"]
        # if args.model == "ae":
        #     model_class = Autoencoder
        #     model_dir_name.append("ae")
        # elif args.model == "sae":
        #     model_class = StackedAutoencoder
        #     model_dir_name.append("sae")
        # elif args.model == "vae":
        #     model_class = VariationalAutoencoder
        #     model_dir_name.append("vae")
        # elif args.model == "svae":
        #     model_class = StackedVAE
        #     model_dir_name.append("svae")
        # else:
        #     print("Unknown model's name {0}, quitting...".format(args.model))
        #     sys.exit()

        if args.prefix:
            model_dir_name.append(args.prefix)

        if args.load:
            print("Loading the model from", args.load)
            model = load(args.load, map_location=lambda storage, loc: storage)
        else:
            print("You must provide both --inference and --load arguments. Quitting...")
            sys.exit()
        print(model.description(), "\n")

        if USE_CUDA:
            model.cuda()


        model_dir_name = (args.output + "/" + "_".join(map(str, model_dir_name)) + "/").replace("//", "/")
        if not os.path.exists(model_dir_name):
            print("Creating new input directory ", model_dir_name, end = "\t")
            os.makedirs(model_dir_name)
        else:
            print("Rewriting the input directory", model_dir_name, end = "\t")
        print("OK.")
        print()

        save_embed(model_dir_name + "/embeddings.txt", model, alphabet, char_indices, seq_vec, len_vec, 1024)

        print("Inference done in", time_since(start)[0] + ".\n")
        sys.exit()



    ##################
    ###  Training  ###
    ##################
    def load_sequence_data(filepath):
        print("Loading the sequence data...", end="\t")
        start = time.time()
        seq_vec = pd.read_table(filepath)["cdr3aa"] + '$'
        len_vec = seq_vec.apply(len).values

        seq_vec = seq_vec.values
        all_seq_num = len_vec.shape[0]
        print("Done in", time_since(start)[0] + ".")
        print(" -- #sequences in the data:", len(seq_vec))

        sorted_inds = sorted(enumerate(len_vec), key=lambda x: x[1], reverse=True)
        seq_vec = seq_vec[[x[0] for x in sorted_inds]]
        len_vec = len_vec[[x[0] for x in sorted_inds]]

        return seq_vec, len_vec

    np.random.seed(42)

    seq_train, len_train = load_sequence_data(args.input_train)
    seq_test, len_test = load_sequence_data(args.input_test)

    all_seq_num = len(seq_train) + len(seq_test)

    # do_megabatches = True
    # if args.megabatch_size >= args.train_size:
    #     args.megabatch_size = args.train_size
    #     do_megabatches = False

    # print(" -- #sequences for training:", len(seq_train), 
    #     "( 100% /", 
    #     str(round(len(seq_train) / all_seq_num * 100, 2)) + "%", ")")
    # print(" -- #sequences per mega-batch:", args.megabatch_size, 
    #     str(round(args.megabatch_size / all_seq_num * 100, 2)) + "%", ")")
    # print(" -- #sequences for testing:", len(seq_test), 
    #     str(round(len(seq_test) / all_seq_num * 100, 2)) + "%", ")")
    # print()


    print("Vectorizing data...", end="\t")
    start = time.time()
    # all_train_indices = [i for i in range(len(seq_train))]
    # train_indices = np.random.choice(all_train_indices, args.megabatch_size, replace=False)
    # test_indices = np.array([i for i in range(len(seq_test))])
    X_train, _, len_train, = vectorize_torch(seq_train, len_train, alphabet, char_indices)
    X_test, _, len_test, = vectorize_torch(seq_test, len_test, alphabet, char_indices)
    
    X_test = X_test.contiguous()
    test_data = TestDatasetSiamese(X_test, seq_test, len_test, 512 if X_test.size(1) > 512 else 4)
    print("Done in", time_since(start)[0] + ".")

    print(" -- X train shape:", X_train.size())
    print(" -- X test shape:", X_test.size(), "\n")
    print("Number of batches to test:", len(test_data))

    model_dir_name = ["siam"]
    if args.model == "ae":
        model_class = Autoencoder
        model_dir_name.append("ae")
    elif args.model == "sae":
        model_class = StackedAutoencoder
        model_dir_name.append("sae")
    elif args.model == "vae":
        model_class = VariationalAutoencoder
        model_dir_name.append("vae")
    elif args.model == "svae":
        model_class = StackedVAE
        model_dir_name.append("svae")
    else:
        print("Unknown model's name {0}, quitting...".format(args.model))
        sys.exit()

    model_dir_name.append("lat" + str(args.latent_dim))
    model_dir_name.append("hid" + str(args.hidden_dim))
    if args.prefix:
        model_dir_name.append(args.prefix)

    if args.load:
        print("Loading the model from", args.load)
        model = load(args.load)
    else:
        model = model_class(len(alphabet), 
            args.latent_dim, 
            args.hidden_dim, 
            args.layers, 
            tuple(map(float, str(args.dropout).split(","))))
        model = SiameseNetwork(model, args.regression_dim, args.regression_layers, tuple(map(float, str(args.dropout).split(",")))[1])
        model.init_weights()
    print(model.description(), "\n")

    if USE_CUDA:
        model.cuda()

    criterion = SiameseNetwork.criterion(args.loss_type, args.target_type)
    me_criterion = SiameseNetwork.mean_error_norep(args.loss_type, args.target_type)

    # criterion = model.criterion_rep(args.loss_type, args.target_type)
    # me_criterion = model.mean_error_rep(args.loss_type, args.target_type)

    loss_log = {"loss_train": [], "loss_test": [], "mean_test": []}
    loss_files = {}
    if args.model in ["vae", "svae"]:
        loss_log["loss_train_rec"] = []
        loss_log["loss_train_kl"] = []
        loss_log["loss_test_rec"] = []
        loss_log["loss_test_kl"] = []


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
    training_start = time.time()
    for i_epoch in range(args.epochs):
        print("Epoch:", i_epoch+1, "/", args.epochs, "[", model_dir_name, "]")

        # if do_megabatches and ((i_epoch + 1) % args.epochs_per_megabatch == 0):
        #     print("Re-vectorizing data...", end="\t")
        #     vector_start = time.time()
        #     train_indices = np.random.choice(all_train_indices, args.megabatch_size, replace=False)
        #     X_train, _, len_train = vectorize_torch(seq_train, len_vec_train, alphabet, char_indices)
        #     print("Done in", time_since(vector_start)[0] + ".")

        learning_rate = args.learning_rate / (5 ** (i_epoch // 30))
        print("lr:", learning_rate)
        optimizer = RMSprop(model.parameters(), lr = learning_rate)

        cur_loss = {x:[] for x in loss_log}

        if args.balancing_mode:
            for i_batch, ((x_left, x_right, lev_dist), (x_single, zero_dist)) in enumerate(batchify_siamese_balanced(X_train, seq_train, len_train, args.batch_size)):
                model.zero_grad()
                optimizer.zero_grad()

                y_pred = model(x_left, x_right)
                loss = list(criterion(y_pred, lev_dist))

                y_pred = model(x_single, x_single)
                loss_zero = criterion(y_pred, zero_dist)
                for i in range(len(loss)):
                    loss[i] += loss_zero[i]

                loss[0].backward()
                optimizer.step()

                train_loss_dict = wrap_loss(loss, WRAP_TRAIN)
                for key, val in train_loss_dict.items():
                    cur_loss[key].append(val)
        else:
            for i_batch, (x_left, x_right, lev_dist) in enumerate(batchify_siamese(X_train, seq_train, len_train, args.batch_size)):
                model.zero_grad()
                optimizer.zero_grad()

                y_pred = model(x_left, x_right)
                loss = criterion(y_pred, lev_dist)

                loss[0].backward()
                optimizer.step()

                train_loss_dict = wrap_loss(loss, WRAP_TRAIN)
                for key, val in train_loss_dict.items():
                    cur_loss[key].append(val)

        tstr, sec = time_since(training_start)

        # Compute the test loss
        test_start = time.time()
        test_loss_dict = test_loss_siamese(model, criterion, me_criterion, test_data)
        for key, val in test_loss_dict.items():
            cur_loss[key].append(val)

        _, test_sec = time_since(test_start)

        print("sec/epoch:", round(sec / (i_epoch+1), 3))
        print("sec/batch:", round(sec / (i_epoch+1) / (i_batch+1), 3), "(" + str(i_batch+1), "batches)")
        print("sec/test: ", round(test_sec, 3))
        for loss_key in sorted(cur_loss.keys()):
            loss_log[loss_key].append(sum(cur_loss[loss_key]))
            print(loss_key, ": ", sum(cur_loss[loss_key]), sep="")

            filename = loss_files[loss_key]
            with open(filename, "a") as outf:
                outf.write(str(sum(cur_loss[loss_key])) + "\n")
        print()

        # test_model(alphabet, model, X_test, y_test, len_test)
        # test_model(alphabet, model, X_train, y_train, len_train)

        # Save latent representation and model weights 
        if ((i_epoch+1) % 10) == 0:
            print("Saving the model...", end="\t")
            save_start = time.time()
            save(model, model_dir_name + "/model." +  str(i_epoch+1) + ".pt")
            print("Done in", time_since(save_start)[0] + ".\n")

        if ((i_epoch+1) % 300) == 0:
            print("Writing latent vectors...", end="\t")
            save_start = time.time()
            # save_embed(model_dir_name + "/embeddings_train." +  str(i_epoch+1) + ".txt", model, alphabet, char_indices, seq_vec[all_train_indices], len_vec[all_train_indices], 1024)
            # save_embed(model_dir_name + "/embeddings_test." +  str(i_epoch+1) + ".txt", model, alphabet, char_indices, seq_test, len_test, 1024)
            print("Done in", time_since(save_start)[0] + ".\n")

    # Save final latent representation and model weights.
    print("Writing final latent vectors and saving the final model...", end="\t")
    save_start = time.time()
    # save_embed(model_dir_name + "/embeddings_train.final.txt", model, alphabet, char_indices, seq_vec[all_train_indices], len_vec[all_train_indices], 1024)
    # save_embed(model_dir_name + "/embeddings_test.final.txt", model, alphabet, char_indices, seq_test, len_test, 1024)

    save(model, model_dir_name + "/model.final.pt")
    print("Done in", time_since(save_start)[0] + ".\n")
