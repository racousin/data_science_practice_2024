import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const TopicModeling = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Topic Modeling</h1>
      <p>
        In this section, you will understand techniques for discovering latent
        topics in text corpora.
      </p>
      <Row>
        <Col>
          <h2>Latent Dirichlet Allocation (LDA)</h2>
          <p>
            LDA is a probabilistic model that assumes each document in a corpus
            is a mixture of topics, and each topic is a distribution over words.
          </p>
          <CodeBlock
            code={`# Example of LDA using Gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
dictionary = Dictionary(corpus)
corpus = [dictionary.doc2bow(doc) for doc in corpus]
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)`}
          />
          <h2>Non-negative Matrix Factorization (NMF)</h2>
          <p>
            NMF is a matrix factorization technique that decomposes a
            document-term matrix into two matrices: one representing the topics
            and the other representing the document-topic distribution.
          </p>
          <CodeBlock
            code={`# Example of NMF using Scikit-learn
from sklearn.decomposition import NMF
nmf = NMF(n_components=10)
W = nmf.fit_transform(X)
H = nmf.components_`}
          />
          <h2>Interpreting and Visualizing Topic Models</h2>
          <p>
            Interpreting and visualizing topic models can help gain insights
            into the underlying structure of the data.
          </p>
          <CodeBlock
            code={`# Example of visualizing LDA topics using pyLDAvis
import pyLDAvis.gensim
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default TopicModeling;
