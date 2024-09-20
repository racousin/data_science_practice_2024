import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Transformer = () => {
  return (
    <Container fluid>
      <h2 id="self-attention">Self-Attention and Multi-Head Attention</h2>
      <p>
        Self-attention, also known as intra-attention, is a mechanism that
        relates different positions of a single sequence to compute a
        representation of the same sequence. It allows the model to attend to
        different parts of the input sequence when encoding each element.
      </p>

      <BlockMath math="\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V" />

      <p>
        Where Q (query), K (key), and V (value) are matrices, and{" "}
        <InlineMath math="d_k" /> is the dimension of the key vectors.
      </p>

      <p>
        Multi-head attention performs the attention function in parallel over
        multiple representation subspaces, allowing the model to jointly attend
        to information from different positions.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention_logits = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        
        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)
        
        return output, attention_weights
        `}
      />

      <h2 id="positional-encoding">Positional Encoding</h2>
      <p>
        Since the Transformer doesn't use recurrence or convolution, it needs
        some way to make use of the order of the sequence. Positional encodings
        are added to the input embeddings to provide information about the
        position of each word in the sequence.
      </p>

      <BlockMath
        math="\begin{aligned}
        PE_{(pos,2i)} &= \sin(pos / 10000^{2i/d_{model}}) \\
        PE_{(pos,2i+1)} &= \cos(pos / 10000^{2i/d_{model}})
      \end{aligned}"
      />

      <CodeBlock
        language="python"
        code={`
import torch
import math

def positional_encoding(max_position, d_model):
    pe = torch.zeros(max_position, d_model)
    position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
        `}
      />

      <h2 id="encoder-decoder">Transformer Encoder and Decoder</h2>
      <p>
        The Transformer uses an encoder-decoder architecture, but replaces the
        recurrent layers with multi-head attention and point-wise, fully
        connected layers.
      </p>

      <h3>Encoder:</h3>
      <ul>
        <li>Self-attention layer</li>
        <li>Feed-forward neural network</li>
      </ul>

      <h3>Decoder:</h3>
      <ul>
        <li>Masked self-attention layer</li>
        <li>Encoder-decoder attention layer</li>
        <li>Feed-forward neural network</li>
      </ul>

      <CodeBlock
        language="python"
        code={`
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Decoder layer would be similar, with an additional encoder-decoder attention layer
        `}
      />

      <h2 id="bert">BERT and its Variants</h2>
      <p>
        BERT (Bidirectional Encoder Representations from Transformers) is a
        transformer-based model designed to pretrain deep bidirectional
        representations from unlabeled text.
      </p>

      <h3>Key features of BERT:</h3>
      <ul>
        <li>Bidirectional training</li>
        <li>Masked Language Model (MLM) pretraining task</li>
        <li>Next Sentence Prediction (NSP) pretraining task</li>
      </ul>

      <p>BERT has spawned many variants, including:</p>
      <ul>
        <li>RoBERTa: Optimized training procedure for BERT</li>
        <li>ALBERT: A Lite BERT with parameter sharing</li>
        <li>DistilBERT: A distilled version of BERT</li>
        <li>
          ELECTRA: Efficiently Learning an Encoder that Classifies Token
          Replacements Accurately
        </li>
      </ul>

      <CodeBlock
        language="python"
        code={`
from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "Hello, how are you?"
encoded_input = tokenizer(text, return_tensors='pt')

# Get BERT embeddings
with torch.no_grad():
    output = model(**encoded_input)

# The pooled output is typically used for classification tasks
pooled_output = output.pooler_output
print(pooled_output.shape)  # [1, 768] for bert-base-uncased
        `}
      />

      <p>
        The Transformer architecture has revolutionized NLP, enabling more
        efficient training on larger datasets and achieving state-of-the-art
        results on a wide range of tasks. BERT and its variants have become the
        foundation for many modern NLP systems.
      </p>
    </Container>
  );
};

export default Transformer;
