import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Representation = () => {
  return (
    <Container fluid>
      <h2 id="bow">Bag of Words (BoW)</h2>
      <p>
        Bag of Words is a simple technique for representing text as numerical
        features. It creates a vocabulary of all unique words in the corpus and
        represents each document as a vector of word counts.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW representation:\n", X.toarray())
        `}
      />

      <h2 id="tfidf">Term Frequency-Inverse Document Frequency (TF-IDF)</h2>
      <p>
        TF-IDF is a statistical measure used to evaluate the importance of a
        word to a document in a collection or corpus. It consists of two parts:
      </p>
      <ul>
        <li>
          Term Frequency (TF): How frequently a word appears in a document
        </li>
        <li>
          Inverse Document Frequency (IDF): How rare or common a word is across
          all documents
        </li>
      </ul>
      <BlockMath math="TF-IDF(t, d, D) = TF(t, d) \times IDF(t, D)" />
      <BlockMath math="IDF(t, D) = \log\frac{N}{|\{d \in D: t \in d\}|}" />
      <p>
        Where t is the term, d is the document, and D is the corpus of
        documents.
      </p>

      <CodeBlock
        language="python"
        code={`
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("TF-IDF representation:\n", X.toarray())
        `}
      />

      <h2 id="word-embeddings">Word Embeddings (Word2Vec, GloVe)</h2>
      <p>
        Word embeddings are dense vector representations of words that capture
        semantic meanings. Unlike BoW or TF-IDF, which are sparse and
        high-dimensional, word embeddings are low-dimensional and can capture
        relationships between words.
      </p>
      <h3>Word2Vec</h3>
      <p>
        Word2Vec uses a neural network model to learn word associations from a
        large corpus of text. It can use two architectures:
      </p>
      <ul>
        <li>
          Continuous Bag of Words (CBOW): Predicts a target word from context
          words
        </li>
        <li>Skip-gram: Predicts context words given a target word</li>
      </ul>

      <CodeBlock
        language="python"
        code={`
from gensim.models import Word2Vec

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)

print("Vocabulary:", list(model.wv.key_to_index.keys()))
print("Vector for 'cat':", model.wv['cat'])
print("Similarity between 'cat' and 'dog':", model.wv.similarity('cat', 'dog'))
        `}
      />

      <h3>GloVe (Global Vectors for Word Representation)</h3>
      <p>
        GloVe is an unsupervised learning algorithm for obtaining vector
        representations for words. It combines the advantages of global matrix
        factorization and local context window methods.
      </p>
      <p>
        While training your own GloVe model requires a large corpus and
        significant computational resources, you can easily use pre-trained
        GloVe embeddings:
      </p>

      <CodeBlock
        language="python"
        code={`
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Convert GloVe file to Word2Vec format
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# Load the pre-trained word vectors
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

print("Vector for 'king':", model['king'])
print("Most similar to 'king':", model.most_similar('king'))
        `}
      />

      <h2 id="subword-embeddings">Subword Embeddings (FastText)</h2>
      <p>
        FastText is an extension of Word2Vec that treats each word as composed
        of character n-grams. This allows the model to generate embeddings for
        out-of-vocabulary words and often performs better for morphologically
        rich languages.
      </p>

      <CodeBlock
        language="python"
        code={`
from gensim.models import FastText

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = FastText(sentences, min_count=1)

print("Vector for 'cat':", model.wv['cat'])
print("Vector for out-of-vocabulary word 'catlike':", model.wv['catlike'])
        `}
      />
    </Container>
  );
};

export default Representation;
