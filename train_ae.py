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


if __name__ == "__main__":
    alphabet = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '$']
    char_indices = dict((c, i) for i, c in enumerate(alphabet))
    indices_char = dict((i, c) for i, c in enumerate(alphabet))

    INPUT_DIM = len(alphabet)


    parser = ArgumentParser()

    #800
    #570
    #500

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
    parser.add_argument("--train_size", "--train", 
        type=int, default=1500000, 
        help="Number of training examples")
    parser.add_argument("--megabatch_size", "--mb", 
        type=int, default=100000, 
        help="Number of training examples per epoch (to fit into GPU memory")
    parser.add_argument("--epochs_per_megabatch", "--epoch_megabatch", 
        type=int, default=50, 
        help="Number of epochs per mega-batch of 'megabatch_size' size")
    parser.add_argument("--test_size", "--test", 
        type=int, default=20000, 
        help="Number of testing examples")
    parser.add_argument("--batch_size", "-b", 
        type=int, default=256, 
        help="Size of the batch")
    parser.add_argument("--epochs", "-e", "--epoch",
        type=int, default=120, 
        help="Number of epochs")
    parser.add_argument("--latent_dim", "--ld", 
        type=int, default=30, 
        help="Number of latent dimensions")
    parser.add_argument("--hidden_dim", "--hd", 
        type=int, default=32, 
        help="GRU hidden layer size")
    parser.add_argument("--layers", "--lay", 
        type=int, default=2, 
        help="Number of layers for GRU")
    parser.add_argument("--dropout", "--drop", 
        type=str, default=".35,.4", 
        help="Dropout for GRU (single value) and linear layers in case of SAE / VAE (comma-separated values)")
    parser.add_argument("--learning_rate", "--lr", 
        type=float, default=1e-3, 
        help="Starting learning rate")
    parser.add_argument("--prefix", "--pr", 
        type=str, default="", 
        help="Prefix to folder")

    args = parser.parse_args()
    # do_megabatches = True
    # if args.megabatch_size >= args.train_size:
    #     args.megabatch_size = args.train_size
    #     do_megabatches = False


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



    ###################
    ###  Inference  ###
    ###################
    # if args.inference:
    #     start = time.time()

    #     seq_vec, len_vec = load_sequence_data(args.inference)

    #     model_dir_name = ["ae", "inference"]

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
    X_train, y_train, len_train = vectorize_torch(seq_train, len_train, alphabet, char_indices)
    X_test, y_test, len_test = vectorize_torch(seq_test, len_test, alphabet, char_indices)
    
    X_test = X_test.contiguous()
    y_test = y_test.contiguous()
    print("Done in", time_since(start)[0] + ".")

    print(" -- X train shape:", X_train.size())
    print(" -- y train shape:", y_train.size())
    print(" -- X test shape:", X_test.size())
    print(" -- y test shape:", y_test.size(), "\n")

    model_dir_name = []
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
    if args.prefix:
        model_dir_name.append(args.prefix)

    if args.load:
        print("Loading the model from", args.load)
        model = load(args.load)
    else:
        model = model_class(len(alphabet), 
            args.latent_dim, 
            args.hidden_dim, args.layers, 
            tuple(map(float, str(args.dropout).split(","))))
        model.init_weights()
    print(model.description(), "\n")

    if USE_CUDA:
        model.cuda()
        

    criterion = model_class.criterion()

    loss_log = {"loss_train": [], "loss_test": []}
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
        #     X_train, y_train, len_train = vectorize_torch(seq_vec, len_vec, alphabet, char_indices, train_indices)
        #     print("Done in", time_since(vector_start)[0] + ".")

        learning_rate = args.learning_rate / (5 ** (i_epoch // 50))
        print("lr:", learning_rate)
        optimizer = RMSprop(model.parameters(), lr = learning_rate)

        cur_loss = {x:[] for x in loss_log}

        for i_batch, (x, y_true, lens) in enumerate(batchify(X_train, y_train, len_train, args.batch_size)):
            model.zero_grad()
            optimizer.zero_grad()

            y_pred = model(x, lens)

            loss = criterion(y_pred, y_true)
            loss[0].backward()
            optimizer.step()

            train_loss_dict = wrap_loss(loss, WRAP_TRAIN)
            for key, val in train_loss_dict.items():
                cur_loss[key].append(val)

        tstr, sec = time_since(training_start)

        # Compute the test loss
        test_start = time.time()
        test_loss_dict = test_loss(model, criterion, X_test, y_test, len_test)
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
        if ((i_epoch+1) % 50) == 0:
            print("Saving the model...", end="\t")
            save_start = time.time()
            save(model, model_dir_name + "/model." +  str(i_epoch+1) + ".pt")
            print("Done in", time_since(save_start)[0] + ".\n")

        if ((i_epoch+1) % 300) == 0:
            print("Writing latent vectors...", end="\t")
            save_start = time.time()
            # save_embed(model_dir_name + "/embeddings_train." +  str(i_epoch+1) + ".txt", model, alphabet, char_indices, seq_vec[all_train_indices], len_vec[all_train_indices], 1024)
            # save_embed(model_dir_name + "/embeddings_test." +  str(i_epoch+1) + ".txt", model, alphabet, char_indices, seq_vec[test_indices], len_vec[test_indices], 1024)
            print("Done in", time_since(save_start)[0] + ".\n")

    # Save final latent representation and model weights.
    print("Writing final latent vectors and saving the final model...", end="\t")
    save_start = time.time()
    # save_embed(model_dir_name + "/embeddings_train.final.txt", model, alphabet, char_indices, seq_test, len_test, 1024)
    # save_embed(model_dir_name + "/embeddings_test.final.txt", model, alphabet, char_indices, seq_test, len_test, 1024)

    save(model, model_dir_name + "/model.final.pt")
    print("Done in", time_since(save_start)[0] + ".\n")