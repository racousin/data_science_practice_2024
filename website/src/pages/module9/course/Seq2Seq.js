import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Seq2Seq = () => {
  return (
    <Container fluid>
      <h2 id="encoder-decoder">Encoder-Decoder Architecture</h2>
      <p>
        Sequence-to-Sequence (Seq2Seq) models are designed to transform an input
        sequence into an output sequence. They consist of two main components:
      </p>
      <ul>
        <li>
          <strong>Encoder:</strong> Processes the input sequence and compresses
          it into a context vector
        </li>
        <li>
          <strong>Decoder:</strong> Takes the context vector and generates the
          output sequence
        </li>
      </ul>
      <p>
        This architecture is particularly useful for tasks where the input and
        output sequences can have different lengths, such as machine translation
        or text summarization.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        encoder_output, encoder_hidden = self.encoder(input_seq)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(target_seq.size(0)):
            decoder_output, decoder_hidden = self.decoder(target_seq[i], decoder_hidden)
            decoder_outputs.append(decoder_output)

        return torch.cat(decoder_outputs, dim=0)

# Example usage
input_size = 1000  # vocabulary size
hidden_size = 256
output_size = 1000

encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)
model = Seq2Seq(encoder, decoder)

input_seq = torch.LongTensor([[1, 2, 3, 4, 5]])  # Example input sequence
target_seq = torch.LongTensor([[6, 7, 8, 9, 10]])  # Example target sequence

output = model(input_seq, target_seq)
print(output.shape)
        `}
      />

      <h2 id="attention">Attention Mechanism</h2>
      <p>
        The attention mechanism allows the model to focus on different parts of
        the input sequence when generating each element of the output sequence.
        This greatly improves the model's ability to handle long sequences and
        capture long-range dependencies.
      </p>

      <BlockMath
        math="\begin{aligned}
        e_{ij} &= a(s_{i-1}, h_j) \\
        \alpha_{ij} &= \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})} \\
        c_i &= \sum_{j=1}^T \alpha_{ij} h_j
      \end{aligned}"
      />

      <p>
        Where <InlineMath math="s_{i-1}" /> is the decoder state,{" "}
        <InlineMath math="h_j" /> is the encoder hidden state,{" "}
        <InlineMath math="a" /> is an alignment model, and{" "}
        <InlineMath math="c_i" /> is the context vector.
      </p>

      <CodeBlock
        language="python"
        code={`
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

# Usage remains similar to the previous example, but with AttentionDecoder instead of Decoder
        `}
      />

      <h2 id="applications">
        Applications (Machine Translation, Text Summarization)
      </h2>
      <h3>Machine Translation</h3>
      <p>
        Seq2Seq models with attention are widely used for machine translation.
        The encoder processes the input sentence in the source language, and the
        decoder generates the translation in the target language.
      </p>

      <h3>Text Summarization</h3>
      <p>
        For text summarization, the encoder processes the full text, and the
        decoder generates a condensed summary. This can be used for both
        extractive summarization (selecting important sentences) and abstractive
        summarization (generating new sentences).
      </p>

      <CodeBlock
        language="python"
        code={`
# Example of using a pre-trained model for translation
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translate English to German
input_text = "Hello, how are you?"
translated = model.generate(**tokenizer(input_text, return_tensors="pt", padding=True))
print(tokenizer.decode(translated[0], skip_special_tokens=True))

# For summarization, you can use models like BART or T5 in a similar way
        `}
      />

      <p>
        Seq2Seq models with attention have been very successful in various NLP
        tasks. However, they have been largely superseded by Transformer-based
        models in recent years, which offer better parallelization and can
        capture long-range dependencies more effectively.
      </p>
    </Container>
  );
};

export default Seq2Seq;
