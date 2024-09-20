import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ClassificationModels = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Classification Models</h1>
      <p>
        In this section, you will learn about various classification models and
        how to implement and evaluate them.
      </p>
      <Row>
        <Col>
          <h2>Logistic Regression</h2>
          <p>
            Logistic regression is a simple and widely used classification
            model. It assumes a logistic relationship between the input features
            and the probability of the target variable.
          </p>
          <CodeBlock
            code={`# Example of logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          />
          <h2>Decision Trees and Random Forests</h2>
          <p>
            Decision trees are a type of classification model that make
            decisions based on a series of if-else conditions. Random forests
            are an ensemble method that combines multiple decision trees to
            improve accuracy and prevent overfitting.
          </p>
          <CodeBlock
            code={`# Example of a decision tree
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          />
          <h2>Support Vector Machines</h2>
          <p>
            Support vector machines (SVMs) are a type of classification model
            that find a hyperplane that separates the data points of different
            classes.
          </p>
          <CodeBlock
            code={`# Example of a support vector machine
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          />
          <h2>Neural Networks</h2>
          <p>
            Neural networks are a type of classification model that are inspired
            by the structure and function of the human brain. They can be used
            to solve complex problems that are difficult to solve with other
            types of models.
          </p>
          <CodeBlock
            code={`# Example of a neural network using Keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=10)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ClassificationModels;
