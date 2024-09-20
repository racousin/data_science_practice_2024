import React from "react";
import { Container } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const EvaluationMetrics = () => {
  return (
    <Container>
      <h1>Evaluation Metrics for Recommendation Systems</h1>

      <section id="introduction">
        <h2>Introduction</h2>
        <p>
          Evaluating recommendation systems is crucial to assess their
          performance and compare different approaches. Various metrics are used
          depending on the type of recommendation task and the specific goals of
          the system.
        </p>
      </section>

      <section id="accuracy-metrics">
        <h2>Accuracy Metrics</h2>

        <h3>Mean Absolute Error (MAE)</h3>
        <p>
          MAE measures the average absolute difference between predicted and
          actual ratings.
        </p>
        <BlockMath>
          {`MAE = \\frac{1}{n} \\sum_{i=1}^n |y_i - \\hat{y}_i|`}
        </BlockMath>

        <h3>Root Mean Square Error (RMSE)</h3>
        <p>RMSE is similar to MAE but gives more weight to large errors.</p>
        <BlockMath>
          {`RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (y_i - \\hat{y}_i)^2}`}
        </BlockMath>

        <CodeBlock
          code={`
import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Usage
y_true = np.array([4, 3, 5, 2, 1])
y_pred = np.array([3.8, 3.2, 4.9, 2.1, 1.2])
print(f"MAE: {mae(y_true, y_pred):.4f}")
print(f"RMSE: {rmse(y_true, y_pred):.4f}")
          `}
          language="python"
        />
      </section>

      <section id="ranking-metrics">
        <h2>Ranking Metrics</h2>

        <h3>Precision@k and Recall@k</h3>
        <p>
          These metrics evaluate the relevance of the top-k recommended items.
        </p>
        <BlockMath>
          {`Precision@k = \\frac{\\text{# of relevant items in top-k}}{k}`}
        </BlockMath>
        <BlockMath>
          {`Recall@k = \\frac{\\text{# of relevant items in top-k}}{\\text{total # of relevant items}}`}
        </BlockMath>

        <h3>Normalized Discounted Cumulative Gain (NDCG)</h3>
        <p>
          NDCG measures the quality of ranking, taking into account the position
          of relevant items.
        </p>
        <BlockMath>{`NDCG@k = \\frac{DCG@k}{IDCG@k}`}</BlockMath>
        <BlockMath>
          {`DCG@k = \\sum_{i=1}^k \\frac{2^{rel_i} - 1}{\\log_2(i+1)}`}
        </BlockMath>

        <CodeBlock
          code={`
import numpy as np

def precision_at_k(y_true, y_pred, k):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    relevant = y_true[y_pred.argsort()[::-1][:k]]
    return np.sum(relevant) / k

def dcg_at_k(y_true, y_pred, k):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    order = y_pred.argsort()[::-1]
    y_true = y_true[order[:k]]
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg_at_k(y_true, y_pred, k):
    dcg = dcg_at_k(y_true, y_pred, k)
    idcg = dcg_at_k(y_true, y_true, k)
    return dcg / idcg

# Usage
y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
k = 3
print(f"Precision@{k}: {precision_at_k(y_true, y_pred, k):.4f}")
print(f"NDCG@{k}: {ndcg_at_k(y_true, y_pred, k):.4f}")
          `}
          language="python"
        />
      </section>

      <section id="diversity-and-novelty">
        <h2>Diversity and Novelty</h2>

        <h3>Intra-List Diversity</h3>
        <p>Measures how different the recommended items are from each other.</p>
        <BlockMath>
          {`ILD = \\frac{1}{|L|(|L|-1)} \\sum_{i \\in L} \\sum_{j \\in L, j \\neq i} d(i,j)`}
        </BlockMath>
        <p>
          Where <InlineMath>d(i,j)</InlineMath> is a distance measure between
          items i and j, and L is the list of recommendations.
        </p>

        <h3>Novelty</h3>
        <p>
          Measures how unexpected or new the recommended items are to a user.
        </p>
        <BlockMath>
          {`Novelty = -\\frac{1}{|L|} \\sum_{i \\in L} \\log_2 p(i)`}
        </BlockMath>
        <p>
          Where <InlineMath>p(i)</InlineMath> is the probability of item i being
          known to the user (often estimated from the training data).
        </p>
      </section>

      <section id="coverage">
        <h2>Coverage</h2>
        <p>
          Coverage measures the proportion of items that the recommender system
          is able to recommend.
        </p>
        <BlockMath>
          {`Coverage = \\frac{|\\text{Unique recommended items}|}{|\\text{All items}|}`}
        </BlockMath>
      </section>

      <section id="user-studies">
        <h2>User Studies</h2>
        <p>
          While offline metrics are useful, user studies provide invaluable
          insights into the real-world performance of recommendation systems.
          These studies often measure:
        </p>
        <ul>
          <li>User satisfaction</li>
          <li>Perceived relevance of recommendations</li>
          <li>System usability</li>
          <li>User engagement and retention</li>
        </ul>
      </section>

      <section id="online-evaluation">
        <h2>Online Evaluation</h2>
        <p>
          A/B testing is a common method for online evaluation of recommendation
          systems. Key metrics in online evaluation include:
        </p>
        <ul>
          <li>Click-through rate (CTR)</li>
          <li>Conversion rate</li>
          <li>User engagement time</li>
          <li>Revenue or other business-specific metrics</li>
        </ul>
      </section>
    </Container>
  );
};

export default EvaluationMetrics;
