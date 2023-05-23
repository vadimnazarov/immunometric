from torch import Tensor, cuda
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

USE_CUDA = cuda.is_available()


def to_var(x):
    if USE_CUDA: 
        x = x.cuda()
    return Variable(x)


def per_char_loss(y_pred, y_true):
    # final_loss = 0
    # for i in range(len(len_vec)):
        # char_inds = to_var(Tensor([i + k*len(len_vec) for k in range(len_vec[i])]).long())
        # final_loss += F.nll_loss(y_pred.index_select(0, char_inds), y_true.index_select(0, char_inds), size_average=False)
    return (F.nll_loss(y_pred, y_true[0], size_average=False), )