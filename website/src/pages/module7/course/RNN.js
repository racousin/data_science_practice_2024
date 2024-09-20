import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const RNN = () => {
  return (
    <Container fluid>
      <h2 id="sequence-modeling">Sequence Modeling and RNN Architecture</h2>

      <p>
        Recurrent Neural Networks (RNNs) are a class of neural networks designed
        to work with sequential data, such as time series or natural language.
      </p>

      <h3>RNN Architecture</h3>
      <p>
        The key idea behind RNNs is to use sequential information. In a
        traditional neural network, all inputs are assumed to be independent.
        For tasks where this is not the case, RNNs introduce the concept of
        memory.
      </p>

      <BlockMath math="h_t = f_W(h_{t-1}, x_t)" />
      <p>
        Where:
        <ul>
          <li>
            <InlineMath math="h_t" /> is the hidden state at time step t
          </li>
          <li>
            <InlineMath math="x_t" /> is the input at time step t
          </li>
          <li>
            <InlineMath math="f_W" /> is a function with parameters W
          </li>
        </ul>
      </p>

      <h3>Simple RNN Implementation in PyTorch</h3>
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
        # x shape: (batch_size, sequence_length, input_size)
        _, hidden = self.rnn(x)
        # hidden shape: (1, batch_size, hidden_size)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
input_size = 10
hidden_size = 20
output_size = 5
sequence_length = 15
batch_size = 32

model = SimpleRNN(input_size, hidden_size, output_size)
input_tensor = torch.randn(batch_size, sequence_length, input_size)
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([32, 5])
        `}
      />

      <h2 id="vanishing-exploding">Vanishing and Exploding Gradients</h2>

      <p>
        One of the main challenges with traditional RNNs is the vanishing and
        exploding gradient problem, which occurs during backpropagation through
        time.
      </p>

      <h3>Vanishing Gradients</h3>
      <p>
        When backpropagating through many time steps, gradients can become very
        small, effectively preventing the network from learning long-term
        dependencies.
      </p>

      <h3>Exploding Gradients</h3>
      <p>
        Conversely, gradients can also become very large, leading to unstable
        learning.
      </p>

      <h3>Solutions</h3>
      <ul>
        <li>Gradient clipping</li>
        <li>Proper weight initialization</li>
        <li>Using gated architectures like LSTM or GRU</li>
      </ul>

      <h2 id="lstm">Long Short-Term Memory (LSTM) Networks</h2>

      <p>
        LSTM networks are a special kind of RNN capable of learning long-term
        dependencies. They were introduced by Hochreiter & Schmidhuber (1997)
        and were refined and popularized by many people in following work.
      </p>

      <h3>LSTM Architecture</h3>
      <p>
        LSTMs have a chain-like structure, but the repeating module has a
        different structure. Instead of having a single neural network layer,
        there are four, interacting in a very special way.
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

      <h3>LSTM Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        _, (hidden, _) = self.lstm(x)
        # hidden shape: (1, batch_size, hidden_size)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
input_size = 10
hidden_size = 20
output_size = 5
sequence_length = 15
batch_size = 32

model = LSTMModel(input_size, hidden_size, output_size)
input_tensor = torch.randn(batch_size, sequence_length, input_size)
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([32, 5])
        `}
      />

      <h2 id="gru">Gated Recurrent Units (GRU)</h2>

      <p>
        GRU is a gating mechanism in recurrent neural networks, introduced in
        2014 by Cho et al. The GRU can be seen as a variation on the LSTM, as it
        has fewer parameters and is simpler to compute and implement.
      </p>

      <h3>GRU Architecture</h3>
      <BlockMath
        math="\begin{aligned}
        z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) \\
        r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) \\
        \tilde{h}_t &= \tanh(W \cdot [r_t * h_{t-1}, x_t]) \\
        h_t &= (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
      \end{aligned}"
      />

      <h3>GRU Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        _, hidden = self.gru(x)
        # hidden shape: (1, batch_size, hidden_size)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
input_size = 10
hidden_size = 20
output_size = 5
sequence_length = 15
batch_size = 32

model = GRUModel(input_size, hidden_size, output_size)
input_tensor = torch.randn(batch_size, sequence_length, input_size)
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([32, 5])
        `}
      />
    </Container>
  );
};

export default RNN;
