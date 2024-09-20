import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const FeatureExtraction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Feature Extraction from Text</h1>
      <p>
        In this section, you will explore methods to convert text into features
        suitable for machine learning models.
      </p>
      <Row>
        <Col>
          <h2>
            Bag of Words (BoW) and Term Frequency-Inverse Document Frequency
            (TF-IDF)
          </h2>
          <p>
            Bag of Words (BoW) and Term Frequency-Inverse Document Frequency
            (TF-IDF) are simple yet effective methods for converting text into
            numerical features.
          </p>
          <h3>Bag of Words (BoW)</h3>
          <p>
            BoW represents text as a collection of unique words, ignoring the
            order and frequency of words.
          </p>
          <CodeBlock
            code={`# Example of BoW using Scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)`}
          />
          <h3>Term Frequency-Inverse Document Frequency (TF-IDF)</h3>
          <p>
            TF-IDF assigns a weight to each word in a document based on its
            frequency and inverse document frequency.
          </p>
          <CodeBlock
            code={`# Example of TF-IDF using Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)`}
          />
          <h2>Word Embeddings: Word2Vec, GloVe</h2>
          <p>
            Word embeddings represent words as dense vectors, capturing semantic
            and syntactic relationships between words.
          </p>
          <h3>Word2Vec</h3>
          <p>
            Word2Vec is a neural network-based method for learning word
            embeddings.
          </p>
          <CodeBlock
            code={`# Example of Word2Vec using Gensim
from gensim.models import Word2Vec
model = Word2Vec(sentences, min_count=1)
word_vectors = model.wv`}
          />
          <h3>GloVe</h3>
          <p>
            GloVe is a method for learning word embeddings based on the global
            statistics of word-word co-occurrence in a corpus.
          </p>
          <CodeBlock
            code={`# Example of GloVe using SpaCy
import spacy
nlp = spacy.load('en_core_web_lg')
word_vectors = nlp.vocab.vectors`}
          />
          <h2>Advanced Embeddings: BERT, ELMo</h2>
          <p>
            Advanced embeddings, such as BERT and ELMo, capture contextual
            information and fine-grained semantic relationships between words.
          </p>
          <h3>BERT</h3>
          <p>
            BERT is a transformer-based method for learning contextual word
            embeddings.
          </p>
          <CodeBlock
            code={`# Example of BERT using Hugging Face Transformers
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')`}
          />
          <h3>ELMo</h3>
          <p>
            ELMo is a method for learning contextual word embeddings based on a
            bidirectional LSTM.
          </p>
          <CodeBlock
            code={`# Example of ELMo using AllenNLP
from allennlp.modules.elmo import Elmo
elmo = Elmo('options.json', 'weights.hdf5', 1)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default FeatureExtraction;
