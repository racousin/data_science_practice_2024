import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const HybridMethods = () => {
  return (
    <Container>
      <h1>Hybrid Methods in Recommendation Systems</h1>

      <section id="definition">
        <h2>Definition</h2>
        <p>
          Hybrid methods in recommendation systems combine multiple
          recommendation techniques to leverage their strengths and mitigate
          their weaknesses. These methods aim to provide more accurate and
          robust recommendations than single-technique approaches.
        </p>
      </section>

      <section id="weighted-hybrid">
        <h2>Weighted Hybrid</h2>
        <p>
          Weighted hybrid methods combine the outputs of different
          recommendation techniques using a weighted sum.
        </p>
        <h3>Mathematical Formulation:</h3>
        <BlockMath>
          {`score(u,i) = w_1 \\cdot score_{CF}(u,i) + w_2 \\cdot score_{CB}(u,i)`}
        </BlockMath>
        <p>Where:</p>
        <ul>
          <li>
            <InlineMath>score(u,i)</InlineMath>: Final recommendation score for
            user u and item i
          </li>
          <li>
            <InlineMath>{`score_{CF}(u,i)`}</InlineMath>: Score from
            collaborative filtering
          </li>
          <li>
            <InlineMath>{`score_{CB}(u,i)`}</InlineMath>: Score from
            content-based filtering
          </li>
          <li>
            <InlineMath>w_1, w_2</InlineMath>: Weights for each method (
            <InlineMath>w_1 + w_2 = 1</InlineMath>)
          </li>
        </ul>
      </section>

      <section id="switching-hybrid">
        <h2>Switching Hybrid</h2>
        <p>
          Switching hybrid methods choose between different recommendation
          techniques based on certain criteria.
        </p>
        <h3>Pseudocode:</h3>
        <CodeBlock
          language="python"
          code={`
def switching_hybrid(user, item):
    if confidence(CF_prediction(user, item)) > threshold:
        return CF_prediction(user, item)
    else:
        return CB_prediction(user, item)

# Where:
# CF_prediction: Collaborative filtering prediction
# CB_prediction: Content-based filtering prediction
# confidence: A function that assesses the reliability of the prediction
# threshold: A predefined confidence threshold
          `}
        />
      </section>

      <section id="feature-combination">
        <h2>Feature Combination</h2>
        <p>
          Feature combination methods incorporate features from different
          recommendation techniques into a single algorithm.
        </p>
        <h3>Example: Combining Content and Collaborative Features</h3>
        <CodeBlock
          language="python"
          code={`
feature_vector = [
    CF_features,  # e.g., user-item interaction patterns
    CB_features   # e.g., item attributes
]

prediction = ML_model(feature_vector)

# Where:
# CF_features: Features extracted from collaborative filtering
# CB_features: Features extracted from content-based filtering
# ML_model: A machine learning model (e.g., neural network, random forest)
          `}
        />
      </section>

      <section id="implementing-hybrid">
        <h2>Implementing Hybrid Methods</h2>
        <p>
          Here's a basic implementation of a weighted hybrid method using
          Python:
        </p>
        <CodeBlock
          language="python"
          code={`
import numpy as np

class WeightedHybridRecommender:
    def __init__(self, cf_model, cb_model, cf_weight=0.7):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.cf_weight = cf_weight
        
    def predict(self, user, item):
        cf_score = self.cf_model.predict(user, item)
        cb_score = self.cb_model.predict(user, item)
        
        return self.cf_weight * cf_score + (1 - self.cf_weight) * cb_score

# Usage
class MockModel:
    def predict(self, user, item):
        return np.random.rand()  # Mock prediction

cf_model = MockModel()
cb_model = MockModel()

hybrid_model = WeightedHybridRecommender(cf_model, cb_model, cf_weight=0.7)
prediction = hybrid_model.predict(user=1, item=2)
print(f"Hybrid prediction: {prediction}")
          `}
        />
      </section>

      <section id="evaluation">
        <h2>Evaluating Hybrid Methods</h2>
        <p>
          Hybrid methods are typically evaluated using standard recommendation
          system metrics, with a focus on comparing their performance to the
          individual methods they combine.
        </p>
        <h3>Common Evaluation Metrics:</h3>
        <ul>
          <li>Mean Absolute Error (MAE)</li>
          <li>Root Mean Square Error (RMSE)</li>
          <li>Precision@k and Recall@k</li>
          <li>Normalized Discounted Cumulative Gain (NDCG)</li>
        </ul>
      </section>
    </Container>
  );
};

export default HybridMethods;
