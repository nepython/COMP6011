import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout_rate, gpu_id=None, **kwargs):
        """
        Define the layers of the model
        Args:
            input_size (int): "Feature" size (in this case, it is 3)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of hidden RNN layers
            n_classes (int): Number of classes in our classification problem
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.gpu_id = gpu_id

        # RNN can be replaced with GRU/LSTM (for GRU the rest of the model stays exactly the same)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True) # batch_first means that the input must have as first dimension the batch size
        # x - > (batch_size, seq_length, input_size) (input of the model)

        self.fc = nn.Linear(hidden_size, n_classes)  # linear layer for the classification part
        # the fully connected layer (fc) only uses the last timestep of the output of the RNN to do the classification

        #self.ol = nn.Sigmoid()

    def forward(self, X, **kwargs):
        """
        Forward Propagation

        Args:
            X: batch of training examples with dimension (batch_size, 1000, 3)
        """
        # initial hidden state:
        h_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.gpu_id)

        out_rnn, _ = self.rnn(X, h_0)
        # out_rnn shape: (batch_size, seq_length, hidden_size) = (batch_size, 1000, hidden_size)

        # decode the hidden state of only the last timestep (other approaches are possible, such as the mean of all states, ..)
        out_rnn = out_rnn[:, -1, :]
        # out_rnn = out_rnn.mean(dim=1)
        # out_rnn shape: (batch_size, hidden_size) - ready to enter the fc layer

        out_fc = self.fc(out_rnn)
        #out = self.ol(out_fc)
        # out shape: (batch_size, num_classes)

        return out_fc


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout_rate, bidirectional, gpu_id=None,
                 **kwargs):
        """
        Define the layers of the model
        Args:
            input_size (int): "Feature" size (in this case, it is 3)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of hidden RNN layers
            n_classes (int): Number of classes in our classification problem
            dropout_rate (float): Dropout rate to apply on all lstm layers except the last one
            bidirectional (bool): Boolean value: if true, lstm layers are bidirectional
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.gpu_id = gpu_id
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True,
                            bidirectional=bidirectional)  # batch_first: first dimension is the batch size

        if bidirectional:
            self.d = 2
        else:
            self.d = 1

        self.fc = nn.Linear(hidden_size*self.d, n_classes)  # linear layer for the classification part

    def forward(self, X, **kwargs):
        """
        Forward Propagation

        Args:
            X: batch of training examples with dimension (batch_size, seq_length, input_size) = (batch_size, 1000, 3)
        """
        # initial hidden state:
        h_0 = torch.zeros(self.num_layers*self.d, X.size(0), self.hidden_size).to(self.gpu_id)
        c_0 = torch.zeros(self.num_layers*self.d, X.size(0), self.hidden_size).to(self.gpu_id)

        out_rnn, _ = self.lstm(X.to(self.gpu_id), (h_0, c_0))
        # out_rnn shape: (batch_size, seq_length, hidden_size*d) = (batch_size, 1000, hidden_size*d)

        if self.bidirectional:
            # concatenate last timestep from the "left-to-right" direction and the first timestep from the
            # "right-to-left" direction
            out_rnn = torch.cat((out_rnn[:, -1, :self.hidden_size], out_rnn[:, 0, self.hidden_size:]), dim=1)
        else:
            # last timestep
            out_rnn = out_rnn[:, -1, :]

        # out_rnn shape: (batch_size, hidden_size*d) - ready to enter the fc layer
        out_fc = self.fc(out_rnn)
        # out_fc shape: (batch_size, num_classes)

        return out_fc


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout_rate, bidirectional, gpu_id=None,
                 **kwargs):
        """
        Define the layers of the model
        Args:
            input_size (int): "Feature" size (in this case, it is 3)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of hidden GRU layers
            n_classes (int): Number of classes in our classification problem
            dropout_rate (float): Dropout rate to be applied in all rnn layers except the last one
            bidirectional (bool): Boolean value: if true, gru layers are bidirectional
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.gpu_id = gpu_id
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        # RNN can be replaced with GRU/LSTM (for GRU the rest of the model stays exactly the same)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True,
                          bidirectional=bidirectional)  # batch_first: first dimension is the batch size

        if bidirectional:
            self.d = 2
        else:
            self.d = 1

        self.fc = nn.Linear(hidden_size*self.d, n_classes)  # linear layer for the classification part

    def forward(self, X, **kwargs):
        """
        Forward Propagation

        Args:
            X: batch of training examples with dimension (batch_size, 1000, 3)
        """
        # initial hidden state:
        h_0 = torch.zeros(self.num_layers*self.d, X.size(0), self.hidden_size).to(self.gpu_id)

        out_rnn, _ = self.rnn(X.to(self.gpu_id), h_0)
        # out_rnn shape: (batch_size, seq_length, hidden_size*d) = (batch_size, 1000, hidden_size*d)

        if self.bidirectional:
            # concatenate last timestep from the "left-to-right" direction and the first timestep from the
            # "right-to-left" direction
            out_rnn = torch.cat((out_rnn[:, -1, :self.hidden_size], out_rnn[:, 0, self.hidden_size:]), dim=1)
        else:
            # last timestep
            out_rnn = out_rnn[:, -1, :]

        # out_rnn shape: (batch_size, hidden_size*d) - ready to enter the fc layer
        out_fc = self.fc(out_rnn)
        # out_fc shape: (batch_size, num_classes)

        return out_fc
