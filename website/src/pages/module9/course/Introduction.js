import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Introduction = () => {
  return (
    <Container fluid>
      <h2 id="what-is-nlp">What is NLP?</h2>
      <p>
        Natural Language Processing (NLP) is a subfield of artificial
        intelligence that focuses on the interaction between computers and
        humans using natural language. The ultimate objective of NLP is to read,
        decipher, understand, and make sense of human languages in a valuable
        way.
      </p>
      <p>
        NLP combines computational linguistics—rule-based modeling of human
        language—with statistical, machine learning, and deep learning models.
        These technologies enable computers to process human language in the
        form of text or voice data and to 'understand' its full meaning,
        complete with the speaker or writer's intent and sentiment.
      </p>

      <h2 id="applications">Applications of NLP</h2>
      <p>NLP has a wide range of applications across various domains:</p>
      <ul>
        <li>
          <strong>Machine Translation:</strong> Automatically translating text
          or speech from one language to another (e.g., Google Translate).
        </li>
        <li>
          <strong>Sentiment Analysis:</strong> Determining the sentiment
          (positive, negative, neutral) of a piece of text (e.g., social media
          monitoring).
        </li>
        <li>
          <strong>Chatbots and Virtual Assistants:</strong> Creating
          conversational agents that can interact with humans (e.g., Siri,
          Alexa).
        </li>
        <li>
          <strong>Text Summarization:</strong> Automatically generating concise
          summaries of longer texts.
        </li>
        <li>
          <strong>Named Entity Recognition (NER):</strong> Identifying and
          classifying named entities (e.g., person names, organizations) in
          text.
        </li>
        <li>
          <strong>Question Answering Systems:</strong> Building systems that can
          automatically answer questions posed in natural language.
        </li>
        <li>
          <strong>Speech Recognition:</strong> Converting spoken language into
          text.
        </li>
        <li>
          <strong>Text Generation:</strong> Generating human-like text for
          various applications (e.g., content creation, chatbots).
        </li>
      </ul>

      <h2 id="challenges">Challenges in NLP</h2>
      <p>
        Despite significant advancements, NLP still faces several challenges:
      </p>
      <ul>
        <li>
          <strong>Ambiguity:</strong> Words and sentences can have multiple
          meanings depending on context.
        </li>
        <li>
          <strong>Language Diversity:</strong> There are thousands of languages
          and dialects worldwide, each with its own rules and nuances.
        </li>
        <li>
          <strong>Contextual Understanding:</strong> Understanding context,
          sarcasm, and implicit meaning in text.
        </li>
        <li>
          <strong>Multimodal Integration:</strong> Combining text with other
          modalities like images or video for comprehensive understanding.
        </li>
        <li>
          <strong>Low-Resource Languages:</strong> Lack of data and resources
          for many languages.
        </li>
        <li>
          <strong>Common Sense Reasoning:</strong> Incorporating common sense
          knowledge into NLP systems.
        </li>
        <li>
          <strong>Bias and Fairness:</strong> Ensuring NLP systems are unbiased
          and fair across different demographics.
        </li>
        <li>
          <strong>Interpretability:</strong> Understanding and explaining the
          decisions made by complex NLP models.
        </li>
      </ul>

      <h3>Example: Simple Text Processing with NLTK</h3>
      <p>
        Here's a basic example of text processing using the Natural Language
        Toolkit (NLTK) in Python:
      </p>
      <CodeBlock
        language="python"
        code={`
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def process_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    return stemmed_tokens

# Example usage
text = "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence."
processed_text = process_text(text)
print(processed_text)
        `}
      />
      <p>
        This example demonstrates basic text processing steps including
        tokenization, stopword removal, and stemming. These are fundamental
        operations in many NLP tasks.
      </p>
    </Container>
  );
};

export default Introduction;
