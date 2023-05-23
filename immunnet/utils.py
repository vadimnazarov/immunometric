import time
import math

import numpy as np
from numpy.random import randint

# import torch
# torch.backends.cudnn.enabled = False

from torch import from_numpy, cuda, Tensor, FloatTensor
from torch import sort as torch_sort
from torch import cat as torch_cat
from torch import max as torch_max
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
# from torch.utils.data import TensorDataset, DataLoader
from editdistance import eval as edit_distance


from torch.nn.modules.module import _addindent
import torch



USE_CUDA = cuda.is_available()

WRAP_TRAIN = "train"
WRAP_TEST = "test"
WRAP_CV = "cv"


def to_var(x):
    if USE_CUDA: 
        x = x.cuda()
    return Variable(x)


def vectorize(seqs, max_len, alphabet, char_indices):
    X = np.zeros((max_len, len(seqs), len(alphabet)), dtype=np.uint8)
    y = np.zeros((max_len, len(seqs)), dtype=np.uint8)
    for i, sequence in enumerate(seqs):
        for t, letter in enumerate(sequence):
            X[t, i, char_indices[letter]] = 1
            y[t, i] = char_indices[letter]
        for t in range(len(sequence), max_len):
            X[t, i, char_indices["X"]] = 1
            y[t, i] = char_indices["X"]
    return X, y


def vectorize_torch(seqs, lens, alphabet, char_indices, train_indices=None, test_indices=None):
    if train_indices is None:
        train_indices = np.arange(0, len(seqs), 1)
    train_indices.sort()

    max_len = lens.max()
    X_train, y_train = vectorize(seqs[train_indices], max_len, alphabet, char_indices)
    X_train = from_numpy(X_train).float()
    y_train = from_numpy(y_train).long()
    len_train = lens[train_indices]
    if USE_CUDA:
        X_train = X_train.cuda()
        y_train = y_train.cuda()

    if test_indices is not None:
        test_indices.sort()

        max_len = lens.max()
        X_test, y_test = vectorize(seqs[test_indices], max_len, alphabet, char_indices)
        X_test = from_numpy(X_test).float()
        y_test = from_numpy(y_test).long()
        len_test = lens[test_indices]

        if USE_CUDA:
            X_test = X_test.cuda()
            y_test = y_test.cuda()

        return X_train, y_train, len_train, X_test, y_test, len_test
    else:
        return X_train, y_train, len_train


def time_since(since):
    now = time.time()
    sec = now - since
    s = sec
    m = math.floor(sec / 60)
    s -= m * 60
    return '%dm%ds' % (m, s), sec


def batchify(X, y, len_vec, batch_size):
    inds = np.random.permutation([x for x in range(X.size(1))])

    #
    # WRONG: sorting after permutation!!!
    #
    inds.sort()

    inds_cuda = from_numpy(inds)
    if USE_CUDA:
        inds_cuda = inds_cuda.cuda()
    for i in range(0, len(inds) // batch_size):
        batch_inds = inds_cuda[i*batch_size : (i+1)*batch_size]
        len_sub_vec = len_vec[i*batch_size : (i+1)*batch_size]
        yield (pack_padded_sequence(to_var(X.index_select(1, batch_inds)), 
                                    len_sub_vec), 
               pack_padded_sequence(to_var(y.index_select(1, batch_inds)), 
                                    len_sub_vec), 
               len_sub_vec)

    if len(inds) % batch_size:
        batch_inds = inds_cuda[-(len(inds) % batch_size) : ]
        len_sub_vec = len_vec[-(len(inds) % batch_size) : ]
        yield (pack_padded_sequence(to_var(X.index_select(1, batch_inds)), 
                                    len_sub_vec), 
               pack_padded_sequence(to_var(y.index_select(1, batch_inds)), 
                                    len_sub_vec), 
               len_sub_vec)


def batchify_straight(data_length, batch_size):
    inds = from_numpy(np.array([x for x in range(data_length)]))
    if USE_CUDA:
        inds = inds.cuda()
    for i in range(0, len(inds) // batch_size):
        yield inds[i*batch_size : (i+1)*batch_size]
    if len(inds) % batch_size:
        yield inds[-(len(inds) % batch_size) : ]


import multiprocessing as mp

def compute_lev(xy):
    return edit_distance(xy[0], xy[1])

def batchify_siamese(X, seq_subvec, len_vec, batch_size):
    query = []
    pool = mp.Pool(4)

    #for i in range(50):
    for i in range(X.size(1) // batch_size + 1):
        inds_left = np.sort(np.random.choice(X.size(1), size = batch_size, replace=True))
        inds_right = np.sort(np.random.choice(X.size(1), size = batch_size, replace=True))

        inds_left_cuda = from_numpy(inds_left)
        inds_right_cuda = from_numpy(inds_right)
        if USE_CUDA:
            inds_left_cuda = inds_left_cuda.cuda()
            inds_right_cuda = inds_right_cuda.cuda()

        query.append((inds_left, inds_right, inds_left_cuda, inds_right_cuda))

    for inds_left, inds_right, inds_left_cuda, inds_right_cuda in query:
        lev_dist_vec = pool.map(compute_lev, [(x,y) for x,y in zip(seq_subvec[inds_left], seq_subvec[inds_right])])
        yield (pack_padded_sequence(to_var(X.index_select(1, inds_left_cuda)), 
                                    len_vec[inds_left]), 
               pack_padded_sequence(to_var(X.index_select(1, inds_right_cuda)), 
                                    len_vec[inds_right]), 
               to_var(Tensor(lev_dist_vec)))


def batchify_siamese_classic(X, seq_subvec, len_vec, batch_size):
    for i in range(X.size(1) // batch_size + 1):
        inds_left = np.sort(np.random.choice(X.size(1), size = batch_size, replace=True))
        inds_right = np.sort(np.random.choice(X.size(1), size = batch_size, replace=True))

        inds_left_cuda = from_numpy(inds_left)
        inds_right_cuda = from_numpy(inds_right)
        if USE_CUDA:
            inds_left_cuda = inds_left_cuda.cuda()
            inds_right_cuda = inds_right_cuda.cuda()

        lev_dist_vec = [0 if x == y else 1 for x,y in zip(seq_subvec[inds_left], seq_subvec[inds_right])]
        yield (pack_padded_sequence(to_var(X.index_select(1, inds_left_cuda)), 
                                    len_vec[inds_left]), 
               pack_padded_sequence(to_var(X.index_select(1, inds_right_cuda)), 
                                    len_vec[inds_right]), 
               to_var(Tensor(lev_dist_vec)))


def batchify_siamese_balanced(X, seq_subvec, len_vec, batch_size):    
    for i in range(X.size(1) // batch_size + 1):
        inds_center = np.sort(np.random.choice(X.size(1), size=batch_size // 4, replace=False))
        inds_left =   np.sort(np.random.choice(X.size(1), size=batch_size * 3 // 4, replace=True))
        inds_right =  np.sort(np.random.choice(X.size(1), size=batch_size * 3 // 4, replace=True))

        inds_center_cuda = from_numpy(inds_center)
        inds_left_cuda = from_numpy(inds_left)
        inds_right_cuda = from_numpy(inds_right)
        if USE_CUDA:
            inds_center_cuda = inds_center_cuda.cuda()
            inds_left_cuda = inds_left_cuda.cuda()
            inds_right_cuda = inds_right_cuda.cuda()

        yield ((pack_padded_sequence(to_var(X.index_select(1, inds_left_cuda)), 
                                    len_vec[inds_left]), 
               pack_padded_sequence(to_var(X.index_select(1, inds_right_cuda)), 
                                    len_vec[inds_right]), 
               to_var(Tensor([edit_distance(x,y) for x,y in zip(seq_subvec[inds_left], seq_subvec[inds_right])]))), 
               (pack_padded_sequence(to_var(X.index_select(1, inds_center_cuda)), 
                                    len_vec[inds_center]), 
               to_var(Tensor([0 for _ in seq_subvec[inds_center]]))))


def batchify_selection(X1, X2, len_vec1, len_vec2, batch_size):
    def get_inds(max_num):
        inds = np.random.permutation(max_num)
        inds_cuda = from_numpy(inds)
        if USE_CUDA:
            inds_cuda = inds_cuda.cuda()
        return inds, inds_cuda

    inds1, inds_cuda1 = get_inds(X1.size(1))
    inds2, inds_cuda2 = get_inds(X2.size(1))

    for i in range(min(X1.size(1), X2.size(1)) // batch_size):
        batch_inds1 = np.sort(inds1[i*batch_size : (i+1)*batch_size])
        batch_inds_cuda1 = inds_cuda1[i*batch_size : (i+1)*batch_size]
        batch_inds_cuda1 = batch_inds_cuda1.sort()[0]

        batch_inds2 = np.sort(inds2[i*batch_size : (i+1)*batch_size])
        batch_inds_cuda2 = inds_cuda2[i*batch_size : (i+1)*batch_size]
        batch_inds_cuda2 = batch_inds_cuda2.sort()[0]

        yield (pack_padded_sequence(to_var(X1.index_select(1, batch_inds_cuda1)), len_vec1[batch_inds1]), 
               pack_padded_sequence(to_var(X2.index_select(1, batch_inds_cuda2)), len_vec2[batch_inds2]))


def batchify_selection_embed(X1, X2, batch_size):
    def get_inds(max_num):
        inds = np.random.permutation(max_num)
        inds_cuda = from_numpy(inds)
        if USE_CUDA:
            inds_cuda = inds_cuda.cuda()
        return inds, inds_cuda

    inds1, inds_cuda1 = get_inds(X1.size(0))
    inds2, inds_cuda2 = get_inds(X2.size(0))

    for i in range(min(X1.size(0), X2.size(0)) // batch_size):
        batch_inds1 = np.sort(inds1[i*batch_size : (i+1)*batch_size])
        batch_inds_cuda1 = inds_cuda1[i*batch_size : (i+1)*batch_size]
        batch_inds_cuda1 = batch_inds_cuda1.sort()[0]

        batch_inds2 = np.sort(inds2[i*batch_size : (i+1)*batch_size])
        batch_inds_cuda2 = inds_cuda2[i*batch_size : (i+1)*batch_size]
        batch_inds_cuda2 = batch_inds_cuda2.sort()[0]

        yield (to_var(X1.index_select(0, batch_inds_cuda1)),
               to_var(X2.index_select(0, batch_inds_cuda2)))


def wrap_loss(loss_value, postfix):
    res = {"loss_" + postfix: loss_value[0].item()}
    if len(loss_value) == 3:
        res["loss_" + postfix + "_rec"] = loss_value[1].item()
        res["loss_" + postfix + "_kl"] = loss_value[2].item()
    return res


def wrap_loss2(loss_value, postfix):
    res = {"loss_" + postfix: loss_value[0]}
    if len(loss_value) == 3:
        res["loss_" + postfix + "_rec"] = loss_value[1]
        res["loss_" + postfix + "_kl"] = loss_value[2]
    return res


def test_loss(model, criterion, X_test, y_test, len_test):
    x_test = pack_padded_sequence(to_var(X_test), len_test)
    y_test = pack_padded_sequence(to_var(y_test), len_test)

    model.eval()
    y_pred = model(x_test, len_test)
    model.train()

    loss_value = criterion(y_pred, y_test)

    return wrap_loss(loss_value, WRAP_TEST)


def test_loss_siamese(model, criterion, mean_error_fun, test_data):
    model.eval()
    loss_list = []
    me_list = []
    for i, batch in enumerate(test_data.get_batch()): 
        # print(i, "/", len(test_data))
        if len(batch) == 2: # balanced
            ((x_left, x_right, lev_dist), (x_single, zero_dist)) = batch
            
            y_pred = model(x_single, x_single)
            loss_value = criterion(y_pred, zero_dist)
            loss_list.append(tuple(x.data[0] for x in loss_value))

            loss_value = mean_error_fun(y_pred, zero_dist)
            me_list.append(loss_value.data[0])

            y_pred = model(x_left, x_right)
            loss_value = criterion(y_pred, lev_dist)
            loss_list.append(tuple(x.data[0] for x in loss_value))

            loss_value = mean_error_fun(y_pred, lev_dist)
            me_list.append(loss_value.data[0])
            
        elif len(batch) == 3: # disbalanced
            x_left, x_right, lev_dist = batch

            y_pred = model(x_left, x_right)
            loss_value = criterion(y_pred, lev_dist)
            loss_list.append(tuple(x.data[0] for x in loss_value))

            loss_value = mean_error_fun(y_pred, lev_dist)
            me_list.append(loss_value.data[0])
        else:
            print("Something wrong with batches")
    model.train()

    final_loss_value = list(loss_list[0])
    for i in range(1, len(loss_list)):
        for j in range(len(final_loss_value)):
            final_loss_value[j] += loss_list[i][j]

    res = wrap_loss2(final_loss_value, WRAP_TEST)
    res["mean_test"] = sum(me_list) / len(me_list)
    return res


def test_loss_selection(model, criterion, X_exp_test, len_exp_X_test, X_gen_test, len_gen_X_test):
    total = 0
    correct = 0

    model.eval()

    y_true = to_var(FloatTensor([1 for _ in range(X_exp_test.size(1))] + [0 for _ in range(X_gen_test.size(1))]).view((-1, 1)))

    y_pred1 = model(pack_padded_sequence(to_var(X_exp_test), len_exp_X_test))
    y_pred2 = model(pack_padded_sequence(to_var(X_gen_test), len_gen_X_test))
    y_pred = torch_cat([y_pred1, y_pred2], 0)

    loss_value = criterion(y_pred, y_true)

    total, correct = accuracy(y_pred, y_true, total, correct)

    model.train()

    print("test", correct, total)
    cur_loss = wrap_loss(loss_value, WRAP_TEST)

    cur_loss["acc_test"] = 100 * correct / total

    return cur_loss


def test_loss_selection_embed(model, criterion, X_exp_test, X_gen_test):
    total = 0
    correct = 0

    model.eval()

    y_true = to_var(FloatTensor([1 for _ in range(X_exp_test.size(0))] + [0 for _ in range(X_gen_test.size(0))]).view((-1, 1)))

    y_pred1 = model(to_var(X_exp_test))
    y_pred2 = model(to_var(X_gen_test))
    y_pred = torch_cat([y_pred1, y_pred2], 0)

    loss_value = criterion(y_pred, y_true)

    y = y_pred.data
    y[y >= .5] = 1
    y[y < .5] = 0
    
    y2 = y_true.data.long()
    total += y2.size(0)
    correct += (y.long() == y2).sum()

    model.train()

    print("test", correct, total)
    cur_loss = wrap_loss(loss_value, WRAP_TEST)

    cur_loss["acc_test"] = 100 * correct / total

    return cur_loss


def test_model(alphabet, model, X_test, Y_test, len_test, max_seq_num=3):
    x_test = pack_padded_sequence(to_var(X_test), len_test)
    y_test = pack_padded_sequence(to_var(Y_test), len_test)

    model.eval()
    y_pred = model(x_test, len_test)
    model.train()

    if type(y_pred) is tuple:
        y_pred = y_pred[0]
    y_test, batch_lens = pad_packed_sequence(y_test)

    ss_true = ["" for x in range(max_seq_num)]
    ss_pred = ["" for x in range(max_seq_num)]

    y_pred = pad_packed_sequence(PackedSequence(data=y_pred, batch_sizes=batch_lens), batch_first=True)[0]
    print(y_pred)
    # Sophisticated code for extraction of predicted characters in batches.
    for i_len, len_val in enumerate(len_test[:max_seq_num]):
        for i_char in range(len_val):
            ss_true[i_len] += alphabet[y_test.data[i_char, i_len]]
            print(y_pred.data[i_len, i_char])
            _, index = y_pred.data[i_char, i_len].max(0)
            ss_pred[i_len] += alphabet[index[0]]

    for i, (s1, s2) in enumerate(zip(ss_true, ss_pred)):
        print(s1, "\n", s2, "\n", sep="")


def save_embed(filename, model, alphabet, char_indices, seq_subvec, len_subvec, batch_size):
    model.eval()
    res = []
    with open(filename, "w") as outf:
        for batch_inds in batchify_straight(len(seq_subvec), batch_size):
            batch_inds = batch_inds.cpu()
            x, _, lens = vectorize_torch(seq_subvec, len_subvec, alphabet, char_indices, batch_inds.numpy())
            x = pack_padded_sequence(to_var(x), lens)

            y_pred = model.encode(x)

            if type(y_pred) is tuple: 
                y_pred = y_pred[0]
            res.append(y_pred.data.cpu().numpy())

            for i_seq in range(len(batch_inds)):
                print(seq_subvec[batch_inds[i_seq]], " ".join(map(lambda z: str(round(z, 5)), y_pred[i_seq].data)), file = outf)

    # with open(filename.replace(".txt", ".npz"), "w") as outf:
    arr = np.zeros((len(seq_subvec), res[0].shape[1]))
    i = 0
    for tensor in res:
        for row in tensor:
            arr[i] = row
            i += 1
    print(arr)
    np.savez_compressed(filename.replace(".txt", ".npz"), arr)
    model.train()


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr


def accuracy(y_pred, y_true, total, correct):
    y = y_pred.data
    y[y >= .5] = 1
    y[y < .5] = 0
    
    y2 = y_true.data.long()
    total += y2.size(0)
    correct += (y.long() == y2).double().sum()

    return total, correct
