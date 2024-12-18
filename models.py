import torch
import torch.nn as nn
import numpy as np

from retention import MultiScaleRetention

from ncps.wirings import AutoNCP
from ncps.torch import LTCCell, LTC

class Embed(nn.Module):
    def __init__(self, input_dim, hidden_dim,embed_dim):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim, track_running_stats=False)
        self.embed1 = nn.Sequential(
           # nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU() 
        )

        self.embed2 = nn.Sequential(
           # nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU() 
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embed1(x)
        x = self.dropout(x)
        x = self.embed2(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, output_size=1, n_hidden_layers = 3, nnodes=128):
        super().__init__()
        self.layer1 = nn.Linear(input_size, nnodes)
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            if i+2 == nnodes:
                break
            else:
                self.hidden_layers.append(nn.Linear(nnodes, nnodes))
        self.layerfin = nn.Linear(nnodes, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layerfin(x) 
        return x


class RetNetModel(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, input_dim, latent_dim=32, double_v_dim=False, forward_type='parallel'):
        super(RetNetModel, self).__init__()
        self.forward_type = forward_type
        self.chunk_size = 2
        self.track_embed = Embed(input_dim[0], hidden_dim * 4, hidden_dim)
        self.track_retention = RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim)
        self.track_ffn = MLP(hidden_dim * 16, output_size=latent_dim, n_hidden_layers = 2, nnodes=latent_dim*4)

        self.vertex_embed = Embed(input_dim[1], hidden_dim * 4, hidden_dim)
        self.vertex_retention = RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim)
        self.vertex_ffn = MLP(hidden_dim * 5, output_size=latent_dim, n_hidden_layers = 2, nnodes=latent_dim*4)  # *4 for chunk2. 5 otherwise

        self.jet_ffn = MLP((2 * latent_dim) + 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        track_x, vertex_x, jet_x = x
        batch_size = track_x.shape[0]
        device = track_x.device

        if self.forward_type == 'parallel':
            # Track branch
            track_x = self.track_embed(track_x)
            track_x = self.track_retention(track_x)
            track_x = track_x.view(batch_size, -1)
            track_x = self.track_ffn(track_x)

            # Vertex branch
            vertex_x = self.vertex_embed(vertex_x)
            vertex_x = self.vertex_retention(vertex_x)
            vertex_x = vertex_x.view(batch_size, -1)
            vertex_x = self.vertex_ffn(vertex_x)

            # Concatenate and final layer
            x = torch.concat((track_x, vertex_x, jet_x), dim=1)
            x = self.jet_ffn(x)
            x = self.sigmoid(x)
            return x

        if self.forward_type == 'recurrent':
            track_x = self.track_embed(track_x)
            s_n_1s_track = self._initialize_state(batch_size, self.track_retention.heads, self.track_retention.hidden_dim, self.track_retention.v_dim, device)
            for i in range(track_x.size(1)):
                track_x[:, i:i+1, :], s_n_1s_track = self.track_retention.forward_recurrent(track_x[:, i:i+1, :], s_n_1s_track, i)
            track_x = track_x.view(batch_size, -1)
            track_x = self.track_ffn(track_x)

            # Vertex branch (recurrent)
            vertex_x = self.vertex_embed(vertex_x)
            s_n_1s_vertex = self._initialize_state(batch_size, self.vertex_retention.heads, self.vertex_retention.hidden_dim, self.vertex_retention.v_dim, device)
            for i in range(vertex_x.size(1)):
                vertex_x[:, i:i+1, :], s_n_1s_vertex = self.vertex_retention.forward_recurrent(vertex_x[:, i:i+1, :], s_n_1s_vertex, i)
            vertex_x = vertex_x.view(batch_size, -1)
            vertex_x = self.vertex_ffn(vertex_x)

            # Concatenate and final layer
            x = torch.concat((track_x, vertex_x, jet_x), dim=1)
            x = self.jet_ffn(x)
            x = self.sigmoid(x)
            return x

        
        if self.forward_type == 'chunkwise':
            track_x = self.track_embed(track_x)
            r_n_1s_track = self._initialize_state(batch_size, self.track_retention.heads, self.track_retention.hidden_dim, self.track_retention.v_dim, device)
            track_chunks = []
            for i in range(track_x.size(1) // self.chunk_size):
                track_chunk, r_n_1s_track = self.track_retention.forward_chunkwise(track_x[:, i*self.chunk_size:(i+1)*self.chunk_size, :], r_n_1s_track, i)
                track_chunks.append(track_chunk)
            track_x = torch.concat(track_chunks, dim=1)
            track_x = track_x.view(batch_size, -1)
            track_x = self.track_ffn(track_x)

            # Vertex branch (chunkwise)
            vertex_x = self.vertex_embed(vertex_x)
            r_n_1s_vertex = self._initialize_state(batch_size, self.vertex_retention.heads, self.vertex_retention.hidden_dim, self.vertex_retention.v_dim, device)
            vertex_chunks = []
            for i in range(vertex_x.size(1) // self.chunk_size):
                vertex_chunk, r_n_1s_vertex = self.vertex_retention.forward_chunkwise(vertex_x[:, i*self.chunk_size:(i+1)*self.chunk_size, :], r_n_1s_vertex, i)
                vertex_chunks.append(vertex_chunk)
            vertex_x = torch.concat(vertex_chunks, dim=1)
            vertex_x = vertex_x.view(batch_size, -1)
            vertex_x = self.vertex_ffn(vertex_x)

            # Concatenate and final layer
            x = torch.concat((track_x, vertex_x, jet_x), dim=1)
            x = self.jet_ffn(x)
            x = self.sigmoid(x)
            return x

    def _initialize_state(self, batch_size, heads, hidden_size, v_dim, device):
        return [torch.zeros(batch_size, hidden_size // heads, v_dim // heads).to(device) for _ in range(heads)]


class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """

        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        s_ns = []
        for i in range(self.layers):

            x_n_clone = x_n.clone()  # Clone x_n to prevent in-place modifications
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n_clone), s_n_1s[i], n)
            y_n = o_n + x_n_clone
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n

        return x_n, s_ns


    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i

        return x_i, r_is

class XGB_Model(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, input_dim, latent_dim=32):
        super(XGB_Model, self).__init__()


        self.ffn = MLP(input_dim, output_size=1, n_hidden_layers=hidden_layers, nnodes=nnodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        track_x, vertex_x, jet_x = x
        batch_size = track_x.shape[0]
        device = track_x.device

        track_x = track_x.flatten(1)        # Start to flatten from dim=1
        vertex_x = vertex_x.flatten(1)
        concat_x = torch.concat((track_x, vertex_x, jet_x), dim=1)
        

        concat_x = self.ffn(concat_x)
        concat_x = self.sigmoid(concat_x)
        return concat_x

class MLP_Model(nn.Module):
    def __init__(self, input_dim, hidden_layers = 3, nnodes=128):
        super(MLP_Model, self).__init__()

        self.ffn = MLP(input_dim, output_size=1, n_hidden_layers=hidden_layers, nnodes=nnodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        track_x, vertex_x, jet_x = x
        batch_size = track_x.shape[0]
        device = track_x.device

        track_x = track_x.flatten(1)        # Start to flatten from dim=1
        vertex_x = vertex_x.flatten(1) 
        concat_x = torch.concat((track_x, vertex_x, jet_x), dim=1)

        concat_x = self.ffn(concat_x)
        concat_x = self.sigmoid(concat_x)
        return concat_x

class LTCcell_Model(nn.Module):
    def __init__(self, input_dim, units, sparsity=0.5):
        super(LTCcell_Model, self).__init__()
     
        self.track_wiring = AutoNCP(units, 1, sparsity_level=sparsity)  # 16 units, 1 motor neuron
        self.track_ltc_cell = LTCCell(self.track_wiring, input_dim[0]) #, batch_first=True)  #return_sequences = False
        self.track_ltc_sequence = RNNSequence(self.track_ltc_cell)    ###

        self.vertex_wiring = AutoNCP(units, 1, sparsity_level=sparsity)  # 16 units, 1 motor neuron
        self.vertex_ltc_cell = LTCCell(self.vertex_wiring, input_dim[1])
        self.vertex_ltc_sequence = RNNSequence(self.vertex_ltc_cell)    ###

        self.jet_ffn = MLP((2 * units) + 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, states, elapsed_time=1.0):
        track_x, vertex_x, jet_x = x
        batch_size = track_x.shape[0]
        device = track_x.device

        # Track branch
        track_x = self.track_ltc_sequence(track_x)
        print(track_x.shape())

        # Vertex branch
        vertex_x = self.vertex_ltc_sequence(vertex_x)
        print(track_x.shape())

        x = torch.concat((track_x, vertex_x, jet_x), dim=1)

        x = self.jet_ffn(x)
        x = self.sigmoid(x)
        return x

    def print_model(self,layout='spiral'):
        sns.set_style("white")
        plt.figure(figsize=(6, 4))
        legend_handles = self.wiring.draw_graph(layout=layout,draw_labels=False,  neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()


class LTC_Model(nn.Module):
    def __init__(self, input_dim, units, sparsity=0.5):
        super(LTC_Model, self).__init__()
     
        self.track_wiring = AutoNCP(units, 1, sparsity_level=sparsity)  # 16 units, 1 motor neuron
        self.track_ltc_cell = LTC(input_dim[0], self.track_wiring, batch_first=True) #, batch_first=True)  #return_sequences = False
   

        self.vertex_wiring = AutoNCP(units, 1, sparsity_level=sparsity)  # 16 units, 1 motor neuron
        self.vertex_ltc_cell = LTC(input_dim[1], self.vertex_wiring, batch_first=True)
     

        self.jet_ffn = MLP(27)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, elapsed_time=1.0):
 
        track_states, vertex_states = None, None

        track_x, vertex_x, jet_x = x
        batch_size = track_x.shape[0]
        device = track_x.device
     
        track_x, track_states = self.track_ltc_cell(track_x, track_states)
        track_x = track_x.reshape([len(track_x),len(track_x[0])])
        vertex_x, vertex_states = self.vertex_ltc_cell(vertex_x, vertex_states)
        vertex_x = vertex_x.reshape([len(vertex_x),len(vertex_x[0])])

        x = torch.concat((track_x, vertex_x, jet_x), dim=1)

        x = self.jet_ffn(x)
        x = self.sigmoid(x)

        return x

    def print_model(self,layout='spiral'):
        sns.set_style("white")
        plt.figure(figsize=(6, 4))
        legend_handles = self.wiring.draw_graph(layout=layout,draw_labels=False,  neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()


class RNNSequence(nn.Module):    # this is needed for LTCCell not LTC!
    def __init__(self,rnn_cell):
        super(RNNSequence, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)

        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device)
     
        for t in range(seq_len):
            inputs = x[:, seq_len-1-t]
            output, hidden_state = self.rnn_cell.forward(inputs, hidden_state, elapsed_time=t*1.0)

        return output

