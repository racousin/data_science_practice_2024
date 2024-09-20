import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Preprocessing = () => {
  return (
    <Container fluid>
      <h2 id="tokenization">Tokenization</h2>
      <p>
        Tokenization is the process of breaking down text into smaller units
        called tokens. These tokens can be words, characters, or subwords.
        Tokenization is a crucial first step in many NLP tasks.
      </p>
      <CodeBlock
        language="python"
        code={`
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello world. How are you doing?"
word_tokens = word_tokenize(text)
sent_tokens = sent_tokenize(text)

print("Word tokens:", word_tokens)
print("Sentence tokens:", sent_tokens)
        `}
      />

      <h2 id="lowercasing">Lowercasing and Normalization</h2>
      <p>
        Lowercasing converts all text to lowercase, which can help reduce the
        vocabulary size and treat words like "The" and "the" as the same token.
        Normalization involves tasks like converting numbers to their word
        equivalents or handling special characters.
      </p>
      <CodeBlock
        language="python"
        code={`
text = "The Quick Brown Fox Jumps Over The Lazy Dog"
lowercased_text = text.lower()
print("Lowercased:", lowercased_text)

import re

def normalize_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert numbers to words
    text = re.sub(r'\d+', 'NUM', text)
    return text

normalized_text = normalize_text("I have 3 apples and 5.2 oranges!")
print("Normalized:", normalized_text)
        `}
      />

      <h2 id="stemming">Stemming and Lemmatization</h2>
      <p>
        Stemming and lemmatization are techniques used to reduce words to their
        root form. Stemming uses a crude heuristic process that chops off the
        ends of words, while lemmatization uses a more sophisticated method
        based on analysis of word structure.
      </p>
      <CodeBlock
        language="python"
        code={`
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "easily", "fairly"]

stemmed_words = [stemmer.stem(word) for word in words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print("Stemmed words:", stemmed_words)
print("Lemmatized words:", lemmatized_words)
        `}
      />

      <h2 id="stop-words">Stop Words Removal</h2>
      <p>
        Stop words are common words that often don't contribute much to the
        meaning of a sentence (e.g., "the", "is", "at"). Removing these can help
        reduce noise in the data.
      </p>
      <CodeBlock
        language="python"
        code={`
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

text = "This is an example sentence demonstrating stop word removal."
word_tokens = word_tokenize(text)

filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]

print("Original:", word_tokens)
print("Filtered:", filtered_sentence)
        `}
      />

      <h2 id="regex">Regular Expressions in NLP</h2>
      <p>
        Regular expressions (regex) are powerful tools for pattern matching and
        text manipulation in NLP tasks. They can be used for tasks like finding
        specific patterns, replacing text, or extracting information.
      </p>
      <CodeBlock
        language="python"
        code={`
import re

text = "The email address is example@email.com and the phone number is 123-456-7890."

# Find email addresses
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(email_pattern, text)

# Find phone numbers
phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
phones = re.findall(phone_pattern, text)

print("Emails found:", emails)
print("Phone numbers found:", phones)

# Replace sensitive information
anonymized_text = re.sub(email_pattern, '[EMAIL]', text)
anonymized_text = re.sub(phone_pattern, '[PHONE]', anonymized_text)

print("Anonymized text:", anonymized_text)
        `}
      />

      <p>
        These preprocessing techniques are fundamental in preparing text data
        for further analysis or model training. The choice of which techniques
        to use often depends on the specific NLP task and the characteristics of
        your data.
      </p>
    </Container>
  );
};

export default Preprocessing;
