import torch
import geoopt.manifolds.poincare.math as pm

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix


    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        # config为bertconfig
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.config = config
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)

            # if self.config.use_hyperbolic:
            #     past_key_values = pm.expmap0(past_key_values, c=1)
            #     print(666)
            #     exit()
        else:
            past_key_values = self.embedding(prefix)

            # if self.config.use_hyperbolic:
            #     past_key_values = pm.expmap0(past_key_values, c=1)
            #     print(666)
            #     exit()
        return past_key_values