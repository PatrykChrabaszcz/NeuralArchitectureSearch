# Non standard cells. Probably slower than cuDNN but more flexible

# https://github.com/pytorch/pytorch/blob/27d7182d6c7e223e04166f33d5ec46ef8b510944/torch/nn/modules/rnn.py#L685

# class GRUCell(RNNCellBase):
#     r"""A gated recurrent unit (GRU) cell
#     .. math::
#         \begin{array}{ll}
#         r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
#         z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
#         n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
#         h' = (1 - z) * n + z * h
#         \end{array}
#     where :math:`\sigma` is the sigmoid function.
#     Args:
#         input_size: The number of expected features in the input x
#         hidden_size: The number of features in the hidden state h
#         bias: If `False`, then the layer does not use bias weights `b_ih` and
#             `b_hh`. Default: `True`
#     Inputs: input, hidden
#         - **input** (batch, input_size): tensor containing input features
#         - **hidden** (batch, hidden_size): tensor containing the initial hidden
#           state for each element in the batch.
#     Outputs: h'
#         - **h'**: (batch, hidden_size): tensor containing the next hidden state
#           for each element in the batch
#     Attributes:
#         weight_ih: the learnable input-hidden weights, of shape
#             `(3*hidden_size x input_size)`
#         weight_hh: the learnable hidden-hidden weights, of shape
#             `(3*hidden_size x hidden_size)`
#         bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
#         bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`
#     Examples::
#         >>> rnn = nn.GRUCell(10, 20)
#         >>> input = Variable(torch.randn(6, 3, 10))
#         >>> hx = Variable(torch.randn(3, 20))
#         >>> output = []
#         >>> for i in range(6):
#         ...     hx = rnn(input[i], hx)
#         ...     output.append(hx)
#     """
#
#     def __init__(self, input_size, hidden_size, bias=True):
#         super(GRUCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
#         self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
#         if bias:
#             self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
#             self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
#         else:
#             self.register_parameter('bias_ih', None)
#             self.register_parameter('bias_hh', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, hx):
#         self.check_forward_input(input)
#         self.check_forward_hidden(input, hx)
#         return self._backend.GRUCell(
#             input, hx,
#             self.weight_ih, self.weight_hh,
#             self.bias_ih, self.bias_hh,
#)