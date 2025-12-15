import torch
import torch.nn as nn
import torch.nn.functional as F


class MyBaseRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = "tanh", bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = self._get_nonlinearity(nonlinearity=nonlinearity)

        self.W = nn.Parameter(torch.empty((hidden_size, input_size)))
        nn.init.xavier_uniform_(self.W)
        self.U = nn.Parameter(torch.empty((hidden_size, hidden_size)))
        nn.init.xavier_uniform_(self.U)

        if bias:
            self.hidden_bias = nn.Parameter(torch.empty(hidden_size))
            nn.init.uniform_(self.hidden_bias)
            self.input_bias = nn.Parameter(torch.empty(hidden_size))
            nn.init.uniform_(self.input_bias)
        else:
            self.hidden_bias = None
            self.input_bias = None

    def _get_nonlinearity(self, nonlinearity: str):
        if nonlinearity == 'tanh':
            return torch.tanh
        elif nonlinearity == 'relu':
            return torch.relu
        else:
            raise NotImplementedError("Not supported nonlinearity activation")


    def forward(self, x, prev_hidden=None):
        """
        x: (seq_len, batch_size, input_size)
        prev_hidden: (batch_size, hidden_size)

        return: outputs: (seq_len, batch_size, hidden_size)
        """

        seq_len, batch_size, _ = x.size()
        if prev_hidden is None:
            prev_hidden = torch.zeros(batch_size, self.hidden_size)
        outputs = []
        hidden = prev_hidden

        for t in range(seq_len):
            input_t = x[t]
            hidden = self.nonlinearity(
                F.linear(input_t, self.W, self.input_bias) +
                F.linear(prev_hidden, self.U, self.hidden_bias)
            )
            outputs.append(hidden.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, hidden

def test_my_base_rnn():
    input_size = 5
    hidden_size = 3
    seq_len = 4
    batch_size = 2

    x = torch.randn(seq_len, batch_size, input_size)
    model = MyBaseRNN(input_size, hidden_size, nonlinearity="tanh", bias=True)
    outputs, last_hidden = model(x)

    print("Input shape:", x.shape)
    print("Outputs shape:", outputs.shape)
    print("Last hidden shape:", last_hidden.shape)
    print("Outputs:", outputs)
    print("Last hidden:", last_hidden)

if __name__ == "__main__":
    test_my_base_rnn()