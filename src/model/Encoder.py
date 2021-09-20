import torch
from torch import nn

class Encoder(nn.Module):
    '''
    embedding_dim - The number of expected features in the input embedding x
    hidden_dim - The number of features in the hidden state h
    num_layers - Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together 
                 to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing
                 the final results.
    dropout - introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout
    '''
    def __init__(self, embedding_dim=64, hidden_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.mlp_dim = 1024

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.hidden_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.hidden_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.hidden_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)

        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]

        return final_h
