import React from "react";
import { Container, Alert } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DockerForDataScience = () => {
  return (
    <Container>
      <h1>Docker for Data Science and Machine Learning</h1>

      <section id="data-science-workflows">
        <h2>Containerizing Data Science Workflows</h2>
        <p>
          Docker can significantly improve reproducibility and portability in
          data science projects. Here's an example Dockerfile for a Python-based
          data science environment:
        </p>
        <CodeBlock
          code={`
FROM python:3.8

RUN pip install numpy pandas scikit-learn matplotlib jupyter

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
          `}
          language="dockerfile"
        />
        <p>
          This Dockerfile sets up a Python environment with common data science
          libraries and Jupyter Notebook.
        </p>
      </section>

      <section id="ml-model-deployment">
        <h2>Deploying Machine Learning Models</h2>
        <p>
          Docker simplifies the deployment of machine learning models. Here's an
          example using Flask to serve a model:
        </p>
        <CodeBlock
          code={`
FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
          `}
          language="dockerfile"
        />
        <p>And the corresponding Python script (app.py):</p>
        <CodeBlock
          code={`
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(data['features'])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
          `}
          language="python"
        />
      </section>

      <section id="gpu-support">
        <h2>GPU Support in Docker</h2>
        <p>
          For GPU-accelerated computing, you can use NVIDIA Container Toolkit:
        </p>
        <CodeBlock
          code={`
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Run a GPU-enabled container
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
          `}
          language="bash"
        />
        <Alert variant="info">
          Note: GPU support requires a compatible NVIDIA GPU and drivers on the
          host system.
        </Alert>
      </section>

      <section id="jupyter-notebooks">
        <h2>Running Jupyter Notebooks in Docker</h2>
        <p>To run Jupyter Notebooks in a Docker container:</p>
        <CodeBlock
          code={`
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/datascience-notebook
          `}
          language="bash"
        />
        <p>
          This command runs a Jupyter Notebook server, maps port 8888, and
          mounts the current directory to the container's work directory.
        </p>
      </section>

      <section id="data-pipelines">
        <h2>Building Data Pipelines with Docker</h2>
        <p>
          Docker can be used to create scalable data pipelines. Here's an
          example using Apache Airflow:
        </p>
        <CodeBlock
          code={`
version: '3'
services:
  webserver:
    image: puckel/docker-airflow:1.10.9
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/usr/local/airflow/dags
    environment:
      - LOAD_EX=n
      - EXECUTOR=Local
  postgres:
    image: postgres:9.6
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
          `}
          language="yaml"
        />
        <p>
          This docker-compose file sets up an Airflow webserver and a Postgres
          database for managing data workflows.
        </p>
      </section>

      <section id="best-practices">
        <h2>Best Practices for Data Science with Docker</h2>
        <ul>
          <li>
            Use specific versions for base images and dependencies to ensure
            reproducibility
          </li>
          <li>Leverage multi-stage builds to keep final images small</li>
          <li>Use volumes to persist data and share it between containers</li>
          <li>
            Implement proper security measures, especially when dealing with
            sensitive data
          </li>
          <li>
            Optimize your Dockerfile to leverage caching and reduce build times
          </li>
          <li>
            Use docker-compose for complex setups involving multiple services
          </li>
        </ul>
      </section>

      <section id="conclusion">
        <h2>Conclusion</h2>
        <p>
          Docker provides powerful tools for data scientists and machine
          learning engineers to create reproducible, scalable, and deployable
          data science workflows. By containerizing data science environments,
          ML models, and data pipelines, teams can ensure consistency across
          development and production environments, simplify collaboration, and
          streamline the deployment process.
        </p>
      </section>
    </Container>
  );
};

export default DockerForDataScience;
