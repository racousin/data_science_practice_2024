import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const TextPreprocessing = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Text Pre-processing</h1>
      <p>
        In this section, you will learn techniques for cleaning and preparing
        text data for analysis.
      </p>
      <Row>
        <Col>
          <h2>Text Normalization Techniques</h2>
          <p>
            Text normalization techniques are used to standardize the text data,
            making it easier to analyze.
          </p>
          <h3>Case Conversion</h3>
          <p>
            Converting all text to lowercase or uppercase can help eliminate
            inconsistencies in the data.
          </p>
          <CodeBlock
            code={`# Example of case conversion using Python
text = text.lower()`}
          />
          <h3>Punctuation Removal</h3>
          <p>
            Removing punctuation can help simplify the text data and reduce
            noise.
          </p>
          <CodeBlock
            code={`# Example of punctuation removal using Python
import string
text = text.translate(str.maketrans('', '', string.punctuation))`}
          />
          <h2>Tokenization and Sentence Segmentation</h2>
          <p>
            Tokenization involves breaking down the text into individual words
            or tokens. Sentence segmentation involves breaking down the text
            into individual sentences.
          </p>
          <CodeBlock
            code={`# Example of tokenization using NLTK
import nltk
tokens = nltk.word_tokenize(text)

# Example of sentence segmentation using NLTK
sentences = nltk.sent_tokenize(text)`}
          />
          <h2>Stop Words Removal and Stemming/Lemmatization</h2>
          <p>
            Stop words are common words that do not contribute much to the
            meaning of the text. Removing stop words can help reduce noise and
            improve efficiency. Stemming and lemmatization involve reducing
            words to their root form.
          </p>
          <CodeBlock
            code={`# Example of stop words removal using NLTK
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Example of stemming using NLTK
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

# Example of lemmatization using SpaCy
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
lemmatized_tokens = [token.lemma_ for token in doc]`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default TextPreprocessing;
