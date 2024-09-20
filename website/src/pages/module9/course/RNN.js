import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const RNN = () => {
  return (
    <Container fluid>
      <h2 id="basic-rnn">Basic RNN Architecture</h2>
      <p>
        Recurrent Neural Networks (RNNs) are a class of neural networks designed
        to work with sequential data, making them particularly suitable for NLP
        tasks. The key idea behind RNNs is that they maintain a hidden state
        that can capture information from previous time steps.
      </p>
      <BlockMath math="h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)" />
      <BlockMath math="y_t = W_{hy}h_t + b_y" />
      <p>
        Where <InlineMath math="h_t" /> is the hidden state at time t,{" "}
        <InlineMath math="x_t" /> is the input at time t, and{" "}
        <InlineMath math="y_t" /> is the output at time t.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
input_sequence = torch.randn(1, 5, 10)  # (batch_size, sequence_length, input_size)
output = model(input_sequence)
print(output.shape)
        `}
      />

      <h2 id="lstm">Long Short-Term Memory (LSTM)</h2>
      <p>
        LSTMs are a type of RNN designed to address the vanishing gradient
        problem in standard RNNs. They introduce a memory cell and gating
        mechanisms to control the flow of information.
      </p>
      <BlockMath
        math="\begin{aligned}
        f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
        i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
        \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
        C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t \\
        o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
        h_t &= o_t * \tanh(C_t)
      \end{aligned}"
      />

      <CodeBlock
        language="python"
        code={`
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
model = LSTMModel(input_size=10, hidden_size=20, output_size=5)
input_sequence = torch.randn(1, 5, 10)
output = model(input_sequence)
print(output.shape)
        `}
      />

      <h2 id="gru">Gated Recurrent Unit (GRU)</h2>
      <p>
        GRU is another variant of RNN that aims to solve the vanishing gradient
        problem. It's similar to LSTM but with a simpler architecture, using
        fewer gates.
      </p>
      <BlockMath
        math="\begin{aligned}
        z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) \\
        r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) \\
        \tilde{h}_t &= \tanh(W \cdot [r_t * h_{t-1}, x_t]) \\
        h_t &= (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
      \end{aligned}"
      />

      <CodeBlock
        language="python"
        code={`
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
model = GRUModel(input_size=10, hidden_size=20, output_size=5)
input_sequence = torch.randn(1, 5, 10)
output = model(input_sequence)
print(output.shape)
        `}
      />

      <h2 id="bidirectional">Bidirectional RNNs</h2>
      <p>
        Bidirectional RNNs process the input sequence in both forward and
        backward directions. This allows the network to capture context from
        both past and future states.
      </p>

      <CodeBlock
        language="python"
        code={`
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(hidden)
        return output

# Example usage
model = BiLSTMModel(input_size=10, hidden_size=20, output_size=5)
input_sequence = torch.randn(1, 5, 10)
output = model(input_sequence)
print(output.shape)
        `}
      />

      <p>
        These RNN variants form the backbone of many sequence modeling tasks in
        NLP. While they have been largely superseded by Transformer-based models
        for many tasks, they still remain relevant, especially in scenarios with
        limited computational resources or when dealing with truly sequential
        data where order is crucial.
      </p>
    </Container>
  );
};

export default RNN;
