import torch
import torch.nn as nn

class GraphFourierLayer(nn.Module):
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        self.N = num_nodes
        self.kernel = nn.Parameter(torch.randn(hidden_dim, hidden_dim, self.N))
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, U):
        """
        x: Tensor of shape (batch, N, hidden_dim)
        U: either
           - (batch, N, N) for a distinct eigenbasis per sample, or
           - (N, N)      for a shared eigenbasis across the batch
        """

        # Handle batched vs static U
        if U.dim() == 3:
            # Batched case: U is (B, N, N)
            x_hat  = torch.einsum('bnh,bnm->bmh', x, U)
            x_filt = torch.einsum('bmh,hdn->bdn', x_hat, self.kernel)
            x_out  = torch.einsum('bdn,bnm->bmd', x_filt, U.transpose(1, 2))
        else:
            # Static case: U is (N, N)
            x_hat  = torch.einsum('bnh,nm->bmh', x, U)
            x_filt = torch.einsum('bmh,hdn->bdn', x_hat, self.kernel)
            x_out  = torch.einsum('bdn,nm->bmd', x_filt, U.T)

        x_res = self.linear(x)
        return x_out + x_res


class GFNO(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.time_emb   = nn.Embedding(7, hidden_dim)          
        self.sector_emb = nn.Embedding(num_nodes, hidden_dim)  

        self.layers = nn.ModuleList([
            GraphFourierLayer(hidden_dim, num_nodes)
            for _ in range(num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x, dow_ids, sector_ids, U):
        """
        x: (batch, N, in_features)
        dow_ids: (batch,) day-of-week
        sector_ids: (batch, N) sector index per node
        U: either (batch, N, N) or (N, N)
        """
        h = self.input_proj(x)                     # (B, N, H)
        h = h + self.time_emb(dow_ids).unsqueeze(1) # (B, 1, H) -> broadcast
        h = h + self.sector_emb(sector_ids)        # (B, N, H)

        for layer in self.layers:
            h = layer(h, U)

        return self.output_head(h)  # (B, N, out_features)