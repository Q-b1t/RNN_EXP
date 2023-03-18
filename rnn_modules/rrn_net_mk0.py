import torch.nn as nn

class rrn_net(nn.Module):
  def __init__(self,input_size,num_hidden,num_layers):
    super().__init__()
    # RNN LAYER
    self.rnn = nn.RNN(
        input_size,
        num_hidden,
        num_layers
    )

    # linear layer for output
    self.out = nn.Linear(num_hidden,1)

  def forward(self,x):
    # run through the RNN layer
    y,hidden = self.rnn(x)
    # and the output (linear) layer
    y = self.out(y)

    return y,hidden
