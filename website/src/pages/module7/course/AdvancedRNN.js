import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const AdvancedRNN = () => {
  return (
    <Container fluid>
      <h2 id="bidirectional">Bidirectional RNNs</h2>

      <p>
        Bidirectional RNNs process sequences in both forward and backward
        directions, allowing the network to capture information from both past
        and future contexts.
      </p>

      <h3>Bidirectional RNN Architecture</h3>
      <p>
        A bidirectional RNN consists of two RNNs: one processing the input
        sequence from left to right, and another processing it from right to
        left. The outputs of both RNNs are typically concatenated.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        _, (hidden, _) = self.lstm(x)
        # hidden shape: (2, batch_size, hidden_size)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(hidden)
        return output

# Example usage
model = BidirectionalRNN(input_size=10, hidden_size=20, output_size=5)
input_tensor = torch.randn(32, 15, 10)  # (batch_size, sequence_length, input_size)
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([32, 5])
        `}
      />

      <h2 id="encoder-decoder">Encoder-Decoder Models</h2>

      <p>
        Encoder-Decoder models, also known as Sequence-to-Sequence (Seq2Seq)
        models, are used for tasks where both input and output are sequences of
        variable length.
      </p>

      <h3>Encoder-Decoder Architecture</h3>
      <p>
        The encoder processes the input sequence and compresses the information
        into a context vector. The decoder then uses this context vector to
        generate the output sequence.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        hidden, cell = self.encoder(source)
        
        input = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
            outputs[:, t] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = target[:, t] if teacher_force else top1.squeeze(1)
        
        return outputs

# Example usage
encoder = Encoder(input_size=10, hidden_size=20)
decoder = Decoder(hidden_size=20, output_size=5)
model = Seq2Seq(encoder, decoder)

source = torch.randn(32, 15, 10)  # (batch_size, source_len, input_size)
target = torch.randint(0, 5, (32, 10))  # (batch_size, target_len)
output = model(source, target)
print(output.shape)  # Should print torch.Size([32, 10, 5])
        `}
      />

      <h2 id="attention">Attention Mechanisms</h2>

      <p>
        Attention mechanisms allow the model to focus on different parts of the
        input sequence when producing each part of the output sequence. This has
        been particularly successful in tasks like machine translation and image
        captioning.
      </p>

      <h3>Attention Calculation</h3>
      <BlockMath
        math="\begin{aligned}
        e_{ij} &= a(s_{i-1}, h_j) \\
        \alpha_{ij} &= \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})} \\
        c_i &= \sum_{j=1}^T \alpha_{ij} h_j
      \end{aligned}"
      />

      <p>
        Where <InlineMath math="s_{i-1}" /> is the previous decoder state,{" "}
        <InlineMath math="h_j" /> is the j-th encoder hidden state,{" "}
        <InlineMath math="a" /> is an alignment model, and{" "}
        <InlineMath math="c_i" /> is the context vector.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(1, 0)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = attention
        
        self.lstm = nn.LSTM(hidden_size + output_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        
        embedded = F.relu(input)
        
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.squeeze(1)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        assert (output == hidden).all()
        
        output = torch.cat((output.squeeze(0), weighted), dim=1)
        
        prediction = self.out(output)
        
        return prediction, hidden, cell

# Usage would be similar to the Seq2Seq model, but with the AttentionDecoder
        `}
      />

      <h2 id="transformer">Transformer Architecture</h2>

      <p>
        The Transformer architecture, introduced in the "Attention Is All You
        Need" paper, relies entirely on attention mechanisms to draw global
        dependencies between input and output. It has become the foundation for
        many state-of-the-art models in NLP.
      </p>

      <h3>Key Components of Transformers</h3>
      <ul>
        <li>Self-Attention</li>
        <li>Multi-Head Attention</li>
        <li>Position-wise Feed-Forward Networks</li>
        <li>Positional Encoding</li>
      </ul>

      <p>
        Due to the complexity of the Transformer architecture, a full
        implementation is beyond the scope of this overview. However, PyTorch
        provides a built-in Transformer module that can be used as follows:
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

# Note: You would need to implement PositionalEncoding separately
        `}
      />
    </Container>
  );
};

export default AdvancedRNN;
