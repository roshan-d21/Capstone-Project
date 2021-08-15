import torch
from torch import nn

from model.Encoder import Encoder

from mode.utils import make_mlp


class Discriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, hidden_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0, d_type='local'
    ):
        super(Discriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [hidden_dim, mlp_dim, 1]

        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

        if d_type == 'global':
            mlp_pool_dims = [hidden_dim + embedding_dim, mlp_dim, hidden_dim]
            self.pool_net = PoolingModule(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=hidden_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)

        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(final_h.squeeze(), seq_start_end, traj[0])

        scores = self.real_classifier(classifier_input)
        return scores
