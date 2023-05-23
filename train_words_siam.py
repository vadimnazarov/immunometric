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
from torch.nn import Module, ModuleList, Sequential, Linear, GRU, NLLLoss, BatchNorm1d, Dropout
from torch.nn.init import kaiming_uniform, kaiming_uniform
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

        self.e_final_mu = Linear(self._hidden_dim, self._latent_dim)
        # self.e_final_lv = Linear(self._hidden_dim, self._latent_dim)

        # self.d_linear_in = Linear(self._latent_dim, self._hidden_dim)
        # self.d_drop_in = Dropout(dropout)
        # self.d_final = Linear(self._hidden_dim, input_dim)


    def forward(self, input, len_vec):
        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)
        return self.decode(z), mu, log_var


    def encode(self, input):
        x= self.e_linear_in(input)
        x = F.elu(x)
        x = self.e_drop_in(x)

        x = self.e_linear1(x)
        x = F.elu(x)
        x = self.e_drop1(x)

        # x = self.e_linear2(x)
        # x = F.elu(x)
        # x = self.e_drop2(x)

        mu = self.e_final_mu(x)
        # log_var = self.e_final_lv(x)
        # return mu, log_var
        return mu


    def encode_(self, input):
        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)
        return z


    # def decode(self, input):
    #     x = self.d_linear_in(input)
    #     x = F.elu(x)
    #     x = self.d_drop_in(x)

    #     x = self.d_final(x)
    #     x = F.elu(x)
    #     return x


    def reparametrize(self, mu, log_var):
        eps = to_var(randn(mu.size(0), mu.size(1)))
        return eps.mul(log_var.mul(.5).exp_()).add_(mu)


    def init_weights(self):
        kaiming_uniform(self.e_linear_in.weight)
        kaiming_uniform(self.e_linear1.weight)
        # kaiming_uniform(self.e_linear2.weight)
        kaiming_uniform(self.e_final_mu.weight)
        # kaiming_uniform(self.e_final_lv.weight)

        # kaiming_uniform(self.d_linear_in.weight)
        # kaiming_uniform(self.d_final.weight)
        


class SiameseNetwork(Module):

    def __init__(self, model):
        super(SiameseNetwork, self).__init__()

        self._model = model


    # Predict log(Lev)
    def forward(self, in_left, in_right):
        # mu1, log_var1 = self.encode(in_left)
        # mu2, log_var2 = self.encode(in_right)
        # return mu1, log_var1, mu2, log_var2

        mu1 = self.encode(in_left)
        mu2 = self.encode(in_right)
        return mu1, mu2


    def encode(self, input):
        return self._model.encode(input)


    def encode_(self, input):
        return self._model.encode_(input)


    def init_weights(self):
        self._model.init_weights()


    def criterion():
        # # Regression layers
        # def _criterion_vae_very_old(y_pred, y_true):
        #     y_pred, mu1, log_var1, mu2, log_var2 = y_pred

        #     mu = torch.cat([mu1, mu2], 0)
        #     log_var = torch.cat([log_var1, log_var2], 0)

        #     kl_divergence = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        #     kl_divergence = torch_sum(kl_divergence).mul_(-.5)

        #     rec_loss = F.mse_loss(y_pred.view((-1,)), y_true, size_average=False)

        #     total_loss = rec_loss + kl_divergence
        #     return total_loss, rec_loss, kl_divergence


        # Prediction of log L
        # def _criterion_vae(y_pred, y_true):
        #     mu1, log_var1, mu2, log_var2 = y_pred
        #     mu = torch.cat([mu1, mu2], 0)
        #     log_var = torch.cat([log_var1, log_var2], 0)

        #     kl_divergence = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        #     kl_divergence = torch_sum(kl_divergence).mul_(-.5)

        #     # y_pred = (mu1 - mu2).abs().sum(1).exp()
        #     y_pred = F.pairwise_distance(mu1, mu2)
        #     rec_loss = F.l1_loss(y_pred.view((-1,)), y_true, size_average=False)

        #     total_loss = rec_loss + kl_divergence
        #     return total_loss, rec_loss, kl_divergence


        def _criterion_novae(y_pred, y_true):
            mu1, mu2 = y_pred

            # y_pred = (mu1 - mu2).abs().sum(1).exp()
            
            y_pred = F.cosine_similarity(mu1, mu2)
            rec_loss = F.mse_loss(y_pred.view((-1,)), y_true, size_average=True)

            # y_pred = F.pairwise_distance(mu1, mu2)
            # rec_loss = F.l1_loss(y_pred.view((-1,)), y_true, size_average=True)

            return (rec_loss, )

        return _criterion_novae



    # def criterion_rep(self):
    #     reparametrize = self._model.reparametrize

    #     def _criterion_vae_new(y_pred, y_true):
    #         mu1, log_var1, mu2, log_var2 = y_pred
    #         mu = torch.cat([mu1, mu2], 0)
    #         log_var = torch.cat([log_var1, log_var2], 0)

    #         kl_divergence = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    #         kl_divergence = torch_sum(kl_divergence).mul_(-.5)

    #         z1 = reparametrize(mu1, log_var1)
    #         z2 = reparametrize(mu2, log_var2)

    #         # y_pred = (z1 - z2).abs().sum(1).exp()
    #         y_pred = F.pairwise_distance(z1, z2)
    #         rec_loss = F.l1_loss(y_pred.view((-1,)), y_true, size_average=False)

    #         total_loss = rec_loss + kl_divergence
    #         return total_loss, rec_loss, kl_divergence

        return _criterion_vae_new


    # Prediction of log L
    def mean_error(y_pred, y_true):
        # mu1, log_var1, mu2, log_var2 = y_pred
        mu1, mu2 = y_pred
        # y_pred = (mu1 - mu2).abs().sum(1).exp()

        # y_pred = F.pairwise_distance(mu1, mu2)
        y_pred = F.cosine_similarity(mu1, mu2)
        rec_loss = F.l1_loss(y_pred.view((-1,)), y_true, size_average=True)
        return rec_loss


    # def mean_error_rep(self):
    #     reparametrize = self._model.reparametrize

    #     def _mean_error(y_pred, y_true):
    #         mu1, log_var1, mu2, log_var2 = y_pred
    #         z1 = reparametrize(mu1, log_var1)
    #         z2 = reparametrize(mu2, log_var2)
    #         # y_pred = (z1 - z2).abs().sum(1).exp()
    #         y_pred = F.pairwise_distance(z1, z2)
    #         rec_loss = F.l1_loss(y_pred.view((-1,)), y_true, size_average=True)
    #         return rec_loss

    #     return _mean_error


    def description(self):
        orig = self._model.description()
        description = """Siamese Network
    Model -> latent vector -> Linear layers -> Prediction
    Number of linear layers: {0}
    Size of linear layers: {1}
    Dropout for linear layers: {2}\n""".format(len(self.reg_layers), self._reg_dim if len(self.reg_layers) > 1 else "-", self._dropout)
        description += "Model: " + orig
        return description


def batchify_words(X, batch_size, n_batches):
    for i in range(n_batches):
        inds1 = from_numpy(np.random.choice(X.size(0), size = batch_size, replace=True))
        inds2 = from_numpy(np.random.choice(X.size(0), size = batch_size, replace=True))

        if USE_CUDA:
            inds1 = inds1.cuda()
            inds2 = inds2.cuda()

        yield (to_var(X[inds1]), 
            to_var(X[inds2]), 
            # to_var(F.pairwise_distance(X[inds1], X[inds2])))
            to_var(F.cosine_similarity(X[inds1], X[inds2])))    


def test_loss_siamese_words(model, criterion, mean_error_fun, test_data):
    model.eval()
    loss_list = []
    me_list = []
    for i, (x_left, x_right, eu_dist) in enumerate(test_data.get_batch()): 
        y_pred = model(x_left, x_right)
        loss_value = criterion(y_pred, eu_dist)
        loss_list.append(tuple(x.data[0] for x in loss_value))

        loss_value = mean_error_fun(y_pred, eu_dist)
        me_list.append(loss_value.data[0])
    model.train()

    final_loss_value = list(loss_list[0])
    for i in range(1, len(loss_list)):
        for j in range(len(final_loss_value)):
            final_loss_value[j] += loss_list[i][j]

    res = wrap_loss2(final_loss_value, WRAP_TEST)
    res["mean_test"] = sum(me_list) / len(me_list)
    return res


class SiameseWordsTestData():

    def __init__(self, X, batch_size, n_batches):
        self.X = X.contiguous()

        if USE_CUDA:
            self.X = self.X.cuda()

        self.bs = batch_size
        self.indices = []

        for i in range(n_batches):
            inds_left =  from_numpy(np.random.choice(self.X.size(0), size=self.bs, replace=True))
            inds_right = from_numpy(np.random.choice(self.X.size(0), size=self.bs, replace=True))

            if USE_CUDA:
                inds_left = inds_left.cuda()
                inds_right = inds_right.cuda()

            self.indices.append((inds_left, inds_right))


    def __len__(self):
        return len(self.indices)


    def get_batch(self):
        for inds_left, inds_right in self.indices:
            yield self._get(inds_left, inds_right)


    def _get(self, inds_left, inds_right):
        return (to_var(self.X[inds_left]), 
                to_var(self.X[inds_right]), 
                # to_var(F.pairwise_distance(self.X[inds_left], self.X[inds_right])))
                to_var(F.cosine_similarity(self.X[inds_left], self.X[inds_right])))


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


    parser = ArgumentParser()

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
        type=float, default=2e-3, 
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
    # if args.inference:
    #     start = time.time()

    #     print("Loading the sequence data...")
    #     seq_vec = pd.read_table(args.inference, sep=",")["cdr3aa"] + '$'
    #     len_vec = seq_vec.apply(len).values

    #     sorted_inds = sorted(enumerate(len_vec), key=lambda x: x[1], reverse=True)
    #     seq_vec = seq_vec[[x[0] for x in sorted_inds]]
    #     len_vec = len_vec[[x[0] for x in sorted_inds]]

    #     model_dir_name = ["siam", "inference"]

    #     if args.prefix:
    #         model_dir_name.append(args.prefix)

    #     if args.load:
    #         print("Loading the model from", args.load)
    #         model = load(args.load, map_location=lambda storage, loc: storage)
    #     else:
    #         print("You must provide both --inference and --load arguments. Quitting...")
    #         sys.exit()
    #     print(model.description(), "\n")

    #     if USE_CUDA:
    #         model.cuda()


    #     model_dir_name = (args.output + "/" + "_".join(map(str, model_dir_name)) + "/").replace("//", "/")
    #     if not os.path.exists(model_dir_name):
    #         print("Creating new input directory ", model_dir_name, end = "\t")
    #         os.makedirs(model_dir_name)
    #     else:
    #         print("Rewriting the input directory", model_dir_name, end = "\t")
    #     print("OK.")
    #     print()

    #     save_embed(model_dir_name + "/embeddings.txt", model, alphabet, char_indices, seq_vec, len_vec, 1024)

    #     print("Inference done in", time_since(start)[0] + ".\n")
    #     sys.exit()



    ##################
    ###  Training  ###
    ##################
    np.random.seed(42)

    print("Loading the data...")
    # wordsAll.npz
    # wordsT.npz
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

    # train_end = 21000
    # test_start = -7000

    # train_end = 9000
    # test_start = -1000

    # train_end = 1000
    # test_start = -1050

    X_test = SiameseWordsTestData(from_numpy(X_train), 2048, 50)
    X_train = from_numpy(X_train).contiguous()
    
    all_names = np.load(words_file)["names"]
    # train_names = all_names[:train_end]
    # test_names = all_names[test_start:]
    train_names = all_names
    test_names = all_names

    # train_names = None
    # test_names = None

    print(" -- X train shape:", X_train.size())

    model_dir_name = ["siam_words"]

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
        model = SiameseNetwork(model)
        model.init_weights()

    if USE_CUDA:
        model.cuda()
        X_train = X_train.cuda()

    criterion = SiameseNetwork.criterion()
    me_criterion = SiameseNetwork.mean_error
    model_dir_name[0] += "_norep_lay2" 

    # criterion = model.criterion_rep()
    # me_criterion = model.mean_error_rep()
    # model_dir_name[0] += "_repar" 

    loss_log = {"loss_train": [], "loss_test": [], "mean_test": []}
    loss_files = {}
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
    for i_epoch in range(args.epochs):
        training_start = time.time()

        print("Epoch:", i_epoch+1, "/", args.epochs, "[", model_dir_name, "]")

        learning_rate = args.learning_rate / (10 ** (i_epoch // 10))
        # learning_rate = args.learning_rate
        # if i_epoch >= 20:
            # learning_rate = args.learning_rate / (5 * (i_epoch // 20))
        print("lr:", learning_rate)
        optimizer = Adam(model.parameters(), lr = learning_rate)

        cur_loss = {x:[] for x in loss_log}

        for i_batch, (x_left, x_right, euc_dist) in enumerate(batchify_words(X_train, args.batch_size, 500)):
            model.zero_grad()
            optimizer.zero_grad()

            y_pred = model(x_left, x_right)
            loss = criterion(y_pred, euc_dist)

            loss[0].backward()
            optimizer.step()

            train_loss_dict = wrap_loss(loss, WRAP_TRAIN)
            for key, val in train_loss_dict.items():
                cur_loss[key].append(val)

        tstr, sec = time_since(training_start)

        # Compute the test loss
        test_start = time.time()
        test_loss_dict = test_loss_siamese_words(model, criterion, me_criterion, X_test)
        for key, val in test_loss_dict.items():
            cur_loss[key].append(val)

        _, test_sec = time_since(test_start)

        print("sec/epoch:", round(sec / (i_epoch+1), 3), "(sec/batch:", round(sec / (i_epoch+1) / (i_batch+1), 3), "[" + str(i_batch+1), "batches])")
        print("sec/test: ", round(test_sec, 3))
        for loss_key in sorted(cur_loss.keys()):
            loss_log[loss_key].append(sum(cur_loss[loss_key]))
            print(loss_key, ": ", sum(cur_loss[loss_key]), sep="")

            filename = loss_files[loss_key]
            with open(filename, "a") as outf:
                outf.write(str(sum(cur_loss[loss_key])) + "\n")
        print()

    #     # test_model(alphabet, model, X_test, y_test, len_test)
    #     # test_model(alphabet, model, X_train, y_train, len_train)

    #     # Save latent representation and model weights 
    #     if ((i_epoch+1) % 50) == 0:
    #         print("Saving the model...", end="\t")
    #         save_start = time.time()
    #         save(model, model_dir_name + "/model." +  str(i_epoch+1) + ".pt")
    #         print("Done in", time_since(save_start)[0] + ".\n")

    #     if ((i_epoch+1) % 300) == 0:
    #         print("Writing latent vectors...", end="\t")
    #         save_start = time.time()
    #         # save_embed(model_dir_name + "/embeddings_train." +  str(i_epoch+1) + ".txt", model, alphabet, char_indices, seq_vec[all_train_indices], len_vec[all_train_indices], 1024)
    #         save_embed(model_dir_name + "/embeddings_test." +  str(i_epoch+1) + ".txt", model, alphabet, char_indices, seq_vec[test_indices], len_vec[test_indices], 1024)
    #         print("Done in", time_since(save_start)[0] + ".\n")

    # Save final latent representation and model weights.
    print("Writing final latent vectors and saving the final model...", end="\t")
    save_start = time.time()
    save_embed_words(model_dir_name + "/embeddings_train.final.txt", model, X_train, train_names, 1024)
    # save_embed_words(model_dir_name + "/embeddings_test.final.txt", model, X_test.X, test_names, 1024)

    save(model, model_dir_name + "/model.final.pt")
    print("Done in", time_since(save_start)[0] + ".\n")