from torch import randn, load
from torch import sum as torch_sum
from torch.nn import Module, Linear, GRU, NLLLoss, BatchNorm1d, Dropout
from torch.nn.init import kaiming_uniform_, kaiming_uniform_

from .utils import *


class StackedVAE(Module):

    def __init__(self, input_dim, latent_dim, rnn_hid_dim, n_layers, dropout):
        super(StackedVAE, self).__init__()

        self._latent_dim = latent_dim
        self._hidden_dim = rnn_hid_dim
        self._n_layers = n_layers
        self._dropout_rnn = dropout[0]
        self._dropout_lin = dropout[1]

        self.e_rnn = GRU(input_dim, self._hidden_dim, n_layers, bidirectional=True, dropout=self._dropout_rnn)
        self.e_linear = Linear(self._hidden_dim, self._hidden_dim)
        # self.e_lin_bn = BatchNorm1d(self._hidden_dim)
        self.e_dropout = Dropout(self._dropout_lin)
        self.e_final_mu = Linear(self._hidden_dim, self._latent_dim)
        self.e_final_lv = Linear(self._hidden_dim, self._latent_dim)

        self.d_rnn = GRU(self._latent_dim, self._hidden_dim, n_layers, bidirectional=True, dropout=self._dropout_rnn)
        self.d_linear = Linear(self._hidden_dim, self._hidden_dim)
        # self.d_lin_bn = BatchNorm1d(self._hidden_dim)
        self.d_dropout = Dropout(self._dropout_lin)
        self.d_final = Linear(self._hidden_dim, input_dim)


    def forward(self, input, len_vec):
        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)
        z = pack_padded_sequence(z.expand(int(len_vec[0]), len(len_vec), self._latent_dim), len_vec)
        return self.decode(z), mu, log_var


    def encode(self, input):
        _, x = self.e_rnn(input)
        x = (x[-1] + x[-2]).mul_(.5).view((-1, self._hidden_dim))
        x = F.elu(x)

        x = self.e_linear(x)
        x = F.elu(x)
        # x = self.e_lin_bn(x)
        x = self.e_dropout(x)

        mu = self.e_final_mu(x)
        log_var = self.e_final_lv(x)
        return mu, log_var


    def decode(self, input):
        output = self.d_rnn(input)[0].data
        x = (output[:, :self._hidden_dim] + output[:, self._hidden_dim:]).mul_(.5)
        x = F.elu(x)

        x = self.d_linear(x)
        x = F.elu(x)
        # x = self.d_lin_bn(x)
        x = self.d_dropout(x)

        x = self.d_final(x)
        x = F.log_softmax(x)
        return x


    def reparametrize(self, mu, log_var):
        eps = to_var(randn(mu.size(0), mu.size(1)))
        return eps.mul(log_var.mul(.5).exp_()).add_(mu)


    def encode_(self, input):
            mu, log_var = self.encode(input)
            z = self.reparametrize(mu, log_var)
            return z


    def init_weights(self):
        kaiming_uniform_(self.e_rnn.weight_ih_l0)
        kaiming_uniform_(self.e_rnn.weight_hh_l0)
        kaiming_uniform_(self.e_linear.weight)
        kaiming_uniform_(self.e_final_mu.weight)
        kaiming_uniform_(self.e_final_lv.weight)

        kaiming_uniform_(self.d_rnn.weight_ih_l0)
        kaiming_uniform_(self.d_rnn.weight_hh_l0)
        kaiming_uniform_(self.d_linear.weight)
        kaiming_uniform_(self.d_final.weight)


    def criterion():
        def vae_loss(y_pred, y_true):
            y_pred, mu, log_var = y_pred

            kl_divergence = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            kl_divergence = torch_sum(kl_divergence).mul_(-.5)

            rec_loss = per_char_loss(y_pred, y_true)[0]

            total_loss = rec_loss + kl_divergence
            return total_loss, rec_loss, kl_divergence

        return vae_loss


    def description(self):
        return """Stacked Variational Autoencoder model
    GRU -> Linear -> Linear -> latent vector x2 -> GRU -> Linear -> Linear 
    Latent vector size: {0}
    GRU hidden size: {1}
    GRU number of layers: {2}
    GRU dropout: {3}
    Hidden Linear size: {1}
    Hidden Linear dropout: {4}""".format(self._latent_dim, self._hidden_dim, self._n_layers, self._dropout_rnn, self._dropout_lin)





