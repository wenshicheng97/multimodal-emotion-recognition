from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=True):
        super(Encoder, self).__init__()

        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        if not activation:
            layers.pop()

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_sizes):
        super(Decoder, self).__init__()

        layers = []
        prev_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_dim, hidden_sizes)
        self.decoder = Decoder(input_dim, hidden_sizes[::-1])

    def forward(self, x):
        return self.decoder(self.encoder(x))


