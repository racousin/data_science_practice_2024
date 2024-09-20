import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DeployingModels = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Deploying Models</h1>
      <p>
        In this section, you will learn about the fundamentals of deploying
        machine learning models to production.
      </p>
      <Row>
        <Col>
          <h2>Introduction to Model Deployment</h2>
          <p>
            Model deployment involves making a model available for use in a
            production environment. This typically involves creating an API that
            can be used to make predictions using the model.
          </p>
          <h2>Deploying a Model with Flask/Django</h2>
          <p>
            Flask and Django are popular frameworks for building web
            applications in Python. They can be used to create a simple API for
            making predictions using a machine learning model.
          </p>
          <CodeBlock
            code={`# Example of deploying a model with Flask
from flask import Flask, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['features'])])
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)`}
          />
          <h2>Monitoring and Maintaining Deployed Models</h2>
          <p>
            Deployed models need to be monitored and maintained to ensure that
            they continue to perform well. This involves monitoring the model's
            performance, re-training the model periodically, and updating the
            model as new data becomes available.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default DeployingModels;
