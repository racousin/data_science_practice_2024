import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const TextClassification = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Text Classification</h1>
      <p>
        In this section, you will master the application of machine learning
        algorithms to classify text data.
      </p>
      <Row>
        <Col>
          <h2>
            Building Classifiers using Naïve Bayes, SVM, and Neural Networks
          </h2>
          <p>
            Naïve Bayes, SVM, and neural networks are popular algorithms for
            text classification.
          </p>
          <h3>Naïve Bayes</h3>
          <p>
            Naïve Bayes is a probabilistic algorithm based on Bayes' theorem,
            assuming that the features are independent.
          </p>
          <CodeBlock
            code={`# Example of Naïve Bayes using Scikit-learn
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)`}
          />
          <h3>SVM</h3>
          <p>
            SVM is a linear model for classification that finds the hyperplane
            that maximizes the margin between classes.
          </p>
          <CodeBlock
            code={`# Example of SVM using Scikit-learn
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)`}
          />
          <h3>Neural Networks</h3>
          <p>
            Neural networks are powerful models for classification that can
            capture complex relationships between features.
          </p>
          <CodeBlock
            code={`# Example of a neural network using TensorFlow
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)`}
          />
          <h2>Sentiment Analysis</h2>
          <p>
            Sentiment analysis is the task of determining the overall sentiment
            (positive, negative, neutral) of a piece of text.
          </p>
          <CodeBlock
            code={`# Example of sentiment analysis using a pre-trained model
from transformers import pipeline
sentiment_pipeline = pipeline('sentiment-analysis')
result = sentiment_pipeline('I love this product!')`}
          />
          <h2>Spam Detection in Emails</h2>
          <p>
            Spam detection in emails involves classifying emails as spam or not
            spam based on their content.
          </p>
          <CodeBlock
            code={`# Example of spam detection using a Naïve Bayes classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
classifier = MultinomialNB()
classifier.fit(X, labels)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default TextClassification;
