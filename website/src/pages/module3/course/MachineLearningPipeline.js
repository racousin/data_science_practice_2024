import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const MachineLearningPipeline = () => {
  return (
    <Container fluid>
      <h1 className="my-4">The Machine Learning Pipeline</h1>
      <p>
        The machine learning pipeline is a systematic approach to developing and
        deploying ML models. It consists of several interconnected stages, each
        crucial for creating effective and reliable models.
      </p>

      <Row>
        <Col md={12}>
          <h2>1. Problem Definition and Data Collection</h2>
          <p>
            <strong>Objective:</strong> Define the problem and acquire relevant
            data
          </p>
          <p>
            <strong>Input:</strong> Business problem, data sources
          </p>
          <p>
            <strong>Output:</strong> Problem statement, success criteria, raw
            dataset(s)
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>
              Determine if machine learning is the appropriate solution and
              Align ML goals with business objectives
            </li>
            <li>Identify appropriate data sources for the problem type</li>
            <li>Ensure data collection adheres to legal standards</li>
            <li>Set up data versioning and storage systems</li>
          </ul>
          <p>
            <strong>Data Types Specifics:</strong>
          </p>
          <ul>
            <li>Tabular: Determine relevant features and target variable</li>
            <li>
              Time Series: Define time granularity and historical data needs
            </li>
            <li>
              Image/Video: Consider resolution, format, and annotation
              requirements
            </li>
            <li>
              Text: Identify language(s) and text sources (e.g., social media,
              documents)
            </li>
          </ul>
          <p>
            <strong>Roles:</strong> Business Analyst, Data Engineer, Domain
            Expert
          </p>
          <p>
            <strong>Tools:</strong> SQL, Hadoop, Apache Kafka, web scraping
            tools
          </p>

          <h2>2. Data Preprocessing and Feature Engineering</h2>
          <p>
            <strong>Objective:</strong> Clean data and create informative
            features
          </p>
          <p>
            <strong>Input:</strong> Raw dataset(s)
          </p>
          <p>
            <strong>Output:</strong> Processed dataset with engineered features
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>Handle missing data, outliers, and inconsistencies</li>
            <li>Normalize or standardize features as needed</li>
            <li>Create domain-specific features</li>
            <li>Apply dimensionality reduction techniques if necessary</li>
          </ul>
          <p>
            <strong>Data Types Specifics:</strong>
          </p>
          <ul>
            <li>
              Tabular: Encode categorical variables, handle numerical features
            </li>
            <li>Time Series: Extract temporal features, handle seasonality</li>
            <li>
              Image/Video: Apply transformations, augmentations, and feature
              extraction
            </li>
            <li>Text: Tokenization, stemming/lemmatization, embeddings</li>
            <li>
              Deep Learning: Data augmentation, transfer learning preparation
            </li>
          </ul>
          <p>
            <strong>Roles:</strong> Data Scientist, Machine Learning Engineer
          </p>
          <p>
            <strong>Tools:</strong> Pandas, NumPy, Scikit-learn, OpenCV, NLTK,
            TensorFlow
          </p>

          <h2>3. Model Selection, Training, and Evaluation</h2>
          <p>
            <strong>Objective:</strong> Select, train, and evaluate appropriate
            ML models
          </p>
          <p>
            <strong>Input:</strong> Processed dataset
          </p>
          <p>
            <strong>Output:</strong> Trained and validated model(s)
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>
              Choose algorithms based on problem type and data characteristics
            </li>
            <li>Split data into training, validation, and test sets</li>
            <li>Implement cross-validation and hyperparameter tuning</li>
            <li>Evaluate model performance using appropriate metrics</li>
            <li>Analyze error patterns and model behavior</li>
          </ul>
          <p>
            <strong>Data Types Specifics:</strong>
          </p>
          <ul>
            <li>
              Tabular: Decision trees, random forests, gradient boosting, neural
              networks
            </li>
            <li>Time Series: ARIMA, Prophet, RNNs (LSTM, GRU)</li>
            <li>Image: CNNs, object detection models (YOLO, R-CNN), GANs</li>
            <li>Text: RNNs, Transformers (BERT, GPT), topic modeling</li>
            <li>Video: 3D CNNs, action recognition models</li>
            <li>
              Deep Learning: Transfer learning, fine-tuning, custom
              architectures
            </li>
          </ul>
          <p>
            <strong>Roles:</strong> Data Scientist, Machine Learning Engineer,
            Research Scientist
          </p>
          <p>
            <strong>Tools:</strong> Scikit-learn, TensorFlow, PyTorch, Keras,
            XGBoost, Hugging Face
          </p>

          <h2>4. Model Interpretability and Explainability</h2>
          <p>
            <strong>Objective:</strong> Understand and explain model decisions
          </p>
          <p>
            <strong>Input:</strong> Trained model, test data
          </p>
          <p>
            <strong>Output:</strong> Model explanations, feature importance
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>Implement appropriate explainability techniques</li>
            <li>Ensure compliance with regulatory requirements</li>
            <li>Communicate insights to stakeholders effectively</li>
          </ul>
          <p>
            <strong>Data Types Specifics:</strong>
          </p>
          <ul>
            <li>Tabular: SHAP values, LIME, partial dependence plots</li>
            <li>
              Time Series: Feature importance over time, counterfactual
              explanations
            </li>
            <li>Image: Saliency maps, GradCAM, concept activation vectors</li>
            <li>
              Text: Attention visualization, influential training examples
            </li>
            <li>
              Deep Learning: Layer-wise relevance propagation, neuron activation
              analysis
            </li>
          </ul>
          <p>
            <strong>Roles:</strong> Data Scientist, Domain Expert
          </p>
          <p>
            <strong>Tools:</strong> SHAP, LIME, Captum, TensorBoard
          </p>

          <h2>5. Deployment, Monitoring, and Maintenance</h2>
          <p>
            <strong>Objective:</strong> Deploy model to production and maintain
            performance
          </p>
          <p>
            <strong>Input:</strong> Validated model
          </p>
          <p>
            <strong>Output:</strong> Deployed model, monitoring system
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>Prepare model for production environment</li>
            <li>Set up necessary infrastructure and APIs</li>
            <li>Implement monitoring for model performance and data drift</li>
            <li>Establish protocols for model updates and retraining</li>
          </ul>
          <p>
            <strong>Data Types Specifics:</strong>
          </p>
          <ul>
            <li>Tabular: Batch vs. real-time prediction setups</li>
            <li>
              Time Series: Sliding window predictions, continuous retraining
            </li>
            <li>
              Image/Video: Optimize for inference speed, handle varying input
              sizes
            </li>
            <li>
              Text: Manage vocabulary updates, handle out-of-vocabulary words
            </li>
            <li>
              Deep Learning: Model compression, hardware acceleration (GPUs,
              TPUs)
            </li>
          </ul>
          <p>
            <strong>Roles:</strong> MLOps Engineer, DevOps Engineer, Data
            Engineer
          </p>
          <p>
            <strong>Tools:</strong> Docker, Kubernetes, MLflow, Kubeflow,
            TensorFlow Serving
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default MachineLearningPipeline;
