import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Classification = () => {
  return (
    <Container fluid>
      <h2 id="sentiment-analysis">Sentiment Analysis</h2>
      <p>
        Sentiment Analysis is the task of determining the emotional tone behind
        a series of words, used to gain an understanding of the attitudes,
        opinions and emotions expressed within an online mention.
      </p>

      <h3>BERT for Sentiment Analysis</h3>
      <CodeBlock
        language="python"
        code={`
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare input
text = "I love this product! It's amazing."
encoded_input = tokenizer(text, return_tensors='pt')

# Make prediction
with torch.no_grad():
    output = model(**encoded_input)

probabilities = torch.softmax(output.logits, dim=1)
sentiment = "Positive" if probabilities[0][1] > probabilities[0][0] else "Negative"
confidence = probabilities[0][1] if sentiment == "Positive" else probabilities[0][0]

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence.item():.4f}")
        `}
      />

      <h2 id="topic-classification">Topic Classification</h2>
      <p>
        Topic Classification involves assigning predefined categories to a given
        text. This can be useful for organizing large collections of documents
        or for content recommendation systems.
      </p>

      <h3>Multi-class Classification with DistilBERT</h3>
      <CodeBlock
        language="python"
        code={`
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load pre-trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define topics
topics = ['Technology', 'Sports', 'Politics', 'Entertainment']

# Prepare input
text = "The new smartphone features a revolutionary AI chip."
encoded_input = tokenizer(text, return_tensors='pt')

# Make prediction
with torch.no_grad():
    output = model(**encoded_input)

probabilities = torch.softmax(output.logits, dim=1)
predicted_topic = topics[torch.argmax(probabilities).item()]
confidence = torch.max(probabilities).item()

print(f"Predicted Topic: {predicted_topic}")
print(f"Confidence: {confidence:.4f}")
        `}
      />

      <h2 id="spam-detection">Spam Detection</h2>
      <p>
        Spam Detection is a binary classification task where the goal is to
        distinguish between legitimate (ham) and unsolicited (spam) messages.
      </p>

      <h3>Spam Detection with Logistic Regression</h3>
      <CodeBlock
        language="python"
        code={`
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume we have a list of texts and labels
texts = ["Buy now!", "Hello, how are you?", "Claim your prize", "Meeting at 3 PM"]
labels = [1, 0, 1, 0]  # 1 for spam, 0 for ham

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Test on a new message
new_message = "Congratulations! You've won a free iPhone!"
prediction = pipeline.predict([new_message])[0]
print(f"The message is classified as: {'Spam' if prediction == 1 else 'Ham'}")
        `}
      />

      <h3>Evaluation Metrics for Classification</h3>
      <p>Common evaluation metrics for classification tasks include:</p>
      <ul>
        <li>Accuracy: Overall correctness of the model</li>
        <li>
          Precision: Proportion of true positive predictions among all positive
          predictions
        </li>
        <li>
          Recall: Proportion of true positive predictions among all actual
          positive instances
        </li>
        <li>F1-score: Harmonic mean of precision and recall</li>
      </ul>

      <BlockMath math="\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}" />
      <BlockMath math="\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}" />
      <BlockMath math="\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}" />

      <p>
        These metrics provide a comprehensive view of a model's performance,
        especially in cases where the classes are imbalanced.
      </p>

      <p>
        Text classification is a fundamental task in NLP with numerous
        applications. The choice of model and approach often depends on the
        specific problem, the amount of available data, and the required
        inference speed.
      </p>
    </Container>
  );
};

export default Classification;
