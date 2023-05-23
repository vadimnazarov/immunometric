from torch.nn import Module, Linear, GRU, NLLLoss, BatchNorm1d, Dropout
from torch.nn.init import kaiming_uniform_, kaiming_uniform_

from .utils import *


class StackedAutoencoder(Module):

    def __init__(self, input_dim, latent_dim, rnn_hid_dim, n_layers, dropout):
        super(StackedAutoencoder, self).__init__()

        self._latent_dim = latent_dim
        self._hidden_dim = rnn_hid_dim
        self._n_layers = n_layers
        self._dropout_rnn = dropout[0]
        self._dropout_lin = dropout[1]

        self.e_rnn = GRU(input_dim, self._hidden_dim, n_layers, bidirectional=True, dropout=self._dropout_rnn)
        self.e_bn_rnn_lin1 = BatchNorm1d(self._hidden_dim)
        self.e_linear1 = Linear(self._hidden_dim, self._hidden_dim)
        self.e_bn_lin1_fin = BatchNorm1d(self._hidden_dim)
        self.e_dropout = Dropout(self._dropout_lin)
        self.e_final = Linear(self._hidden_dim, self._latent_dim)

        self.d_rnn = GRU(self._latent_dim, self._hidden_dim, n_layers, bidirectional=True, dropout=self._dropout_rnn)
        self.d_bn_rnn_lin1 = BatchNorm1d(self._hidden_dim)
        self.d_linear1 = Linear(self._hidden_dim, self._hidden_dim)
        self.d_bn_lin1_fin = BatchNorm1d(self._hidden_dim)
        self.d_dropout = Dropout(self._dropout_lin)
        self.d_final = Linear(self._hidden_dim, input_dim)


    def forward(self, input, len_vec):
        latent_vec = self.encode(input)
        latent_vec = pack_padded_sequence(latent_vec.expand(int(len_vec[0]), len(len_vec), self._latent_dim), len_vec)
        seq = self.decode(latent_vec)
        return seq


    def encode(self, input):
        _, x = self.e_rnn(input)
        x = (x[-1] + x[-2]).mul_(.5).view((-1, self._hidden_dim))
        x = F.elu(x)
        # x = self.e_bn_rnn_lin1(x)

        x = self.e_linear1(x)
        x = F.elu(x)
        # x = self.e_bn_lin1_fin(x)

        x = self.e_dropout(x)

        x = self.e_final(x)
        return x


    def encode_(self, input):
        return self.encode(input)


    def decode(self, input):
        output = self.d_rnn(input)[0].data
        x = (output[:, :self._hidden_dim] + output[:, self._hidden_dim:]).mul_(.5)
        x = F.elu(x)
        # x = self.d_bn_rnn_lin1(x)

        x = self.d_linear1(x)
        x = F.elu(x)
        # x = self.d_bn_lin1_fin(x)

        x = self.d_dropout(x)

        x = self.d_final(x)
        x = F.log_softmax(x)
        return x


    def init_weights(self):
        kaiming_uniform_(self.e_rnn.weight_ih_l0)
        kaiming_uniform_(self.e_rnn.weight_hh_l0)
        kaiming_uniform_(self.e_linear1.weight)
        kaiming_uniform_(self.e_final.weight)

        kaiming_uniform_(self.d_rnn.weight_ih_l0)
        kaiming_uniform_(self.d_rnn.weight_hh_l0)
        kaiming_uniform_(self.d_linear1.weight)
        kaiming_uniform_(self.d_final.weight)


    def criterion():
        return per_char_loss


    def description(self):
        return """Stacked Autoencoder model
    GRU+BN -> Linear+BN -> Linear -> latent vector -> GRU+BN -> Linear+BN -> Linear
    Latent vector size: {0}
    GRU hidden size: {1}
    GRU number of layers: {2}
    GRU dropout: {3}
    Hidden Linear size: {1}
    Hidden Linear dropout: {4}""".format(self._latent_dim, self._hidden_dim, self._n_layers, self._dropout_rnn, self._dropout_lin)





