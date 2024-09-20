import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const TransferLearning = () => {
  return (
    <Container fluid>
      <h2 id="fine-tuning">Fine-tuning Pre-trained Models</h2>
      <p>
        Fine-tuning is the process of taking a pre-trained model and further
        training it on a specific task or dataset. This allows us to leverage
        the knowledge learned by the model on large datasets and adapt it to our
        specific needs.
      </p>

      <h3>Steps for Fine-tuning:</h3>
      <ol>
        <li>Load a pre-trained model</li>
        <li>Replace the final layer(s) with new ones suitable for your task</li>
        <li>Freeze some or all of the pre-trained layers</li>
        <li>Train the model on your dataset</li>
      </ol>

      <CodeBlock
        language="python"
        code={`
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import torch

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Freeze all layers except the classifier
for param in model.bert.parameters():
    param.requires_grad = False

# Prepare your data
texts = ["This is a positive review", "This is a negative review"]
labels = torch.tensor([1, 0])  # 1 for positive, 0 for negative

# Tokenize and encode the data
encoded_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = encoded_data['input_ids']
attention_mask = encoded_data['attention_mask']

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
        `}
      />

      <h2 id="bert-tasks">BERT for Various NLP Tasks</h2>
      <p>BERT can be fine-tuned for a wide variety of NLP tasks, including:</p>
      <ul>
        <li>Text Classification</li>
        <li>Named Entity Recognition (NER)</li>
        <li>Question Answering</li>
        <li>Sentiment Analysis</li>
        <li>Text Summarization</li>
      </ul>

      <h3>Example: Sentiment Analysis with BERT</h3>
      <CodeBlock
        language="python"
        code={`
from transformers import pipeline

# Load pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze sentiment
text = "I love this product! It's amazing."
result = sentiment_analyzer(text)

print(f"Text: {text}")
print(f"Sentiment: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.4f}")
        `}
      />

      <h2 id="other-models">
        Other Transfer Learning Models (GPT, XLNet, RoBERTa)
      </h2>

      <h3>GPT (Generative Pre-trained Transformer)</h3>
      <p>
        GPT models are autoregressive language models trained on a large corpus
        of text. They excel at text generation tasks.
      </p>

      <CodeBlock
        language="python"
        code={`
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Once upon a time"
input_ids = tokenizer.encode(text, return_tensors='pt')

output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
        `}
      />

      <h3>XLNet</h3>
      <p>
        XLNet is a generalized autoregressive pretraining method that overcomes
        the limitations of BERT by using permutation language modeling.
      </p>

      <h3>RoBERTa (Robustly Optimized BERT Approach)</h3>
      <p>
        RoBERTa is an optimized version of BERT with improved training
        methodology, including longer training time and bigger batches.
      </p>

      <CodeBlock
        language="python"
        code={`
from transformers import RobertaForSequenceClassification, RobertaTokenizer

model = RobertaForSequenceClassification.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

text = "RoBERTa is a robustly optimized BERT pretraining approach."
inputs = tokenizer(text, return_tensors='pt')

outputs = model(**inputs)
print(outputs.logits)
        `}
      />

      <h3>Comparing Transfer Learning Models</h3>
      <table className="table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Key Features</th>
            <th>Best For</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>BERT</td>
            <td>Bidirectional, MLM, NSP</td>
            <td>General NLP tasks, especially understanding context</td>
          </tr>
          <tr>
            <td>GPT</td>
            <td>Autoregressive, left-to-right</td>
            <td>Text generation, language modeling</td>
          </tr>
          <tr>
            <td>XLNet</td>
            <td>Permutation language modeling</td>
            <td>Tasks requiring long-range dependencies</td>
          </tr>
          <tr>
            <td>RoBERTa</td>
            <td>Optimized BERT training</td>
            <td>Improved performance on BERT-like tasks</td>
          </tr>
        </tbody>
      </table>

      <p>
        Transfer learning has significantly advanced the state of NLP by
        allowing models to leverage knowledge from large-scale pretraining. This
        has led to impressive performance on a wide range of tasks, even with
        limited task-specific data.
      </p>
    </Container>
  );
};

export default TransferLearning;
