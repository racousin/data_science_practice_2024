import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const CollaborativeFiltering = () => {
  return (
    <Container>
      <h1>Collaborative Filtering</h1>

      <section id="definition">
        <h2>Definition</h2>
        <p>
          Collaborative filtering is a recommendation technique that makes
          predictions about a user's interests by collecting preferences from
          many users. It operates on the assumption that users who agreed in the
          past tend to agree again in the future.
        </p>
      </section>

      <section id="user-based-cf">
        <h2>User-Based Collaborative Filtering</h2>
        <p>
          User-based CF finds users with similar rating patterns to the target
          user and uses their ratings to predict preferences for the target
          user.
        </p>
        <h3>Mathematical Formulation:</h3>
        <BlockMath>
          {`
\\hat{r}_{ui} = \\mu_u + \\frac{\\sum_{v \\in N(u)} sim(u,v) \\cdot (r_{vi} - \\mu_v)}{\\sum_{v \\in N(u)} |sim(u,v)|}
          `}
        </BlockMath>
        <p>Where:</p>
        <ul>
          <li>
            <InlineMath>{`\hat{r}_{ui}}`}</InlineMath>: Predicted rating of user
            u for item i
          </li>
          <li>
            <InlineMath>\mu_u</InlineMath>: Mean rating of user u
          </li>
          <li>
            <InlineMath>sim(u,v)</InlineMath>: Similarity between users u and v
          </li>
          <li>
            <InlineMath>{`r_{vi}`}</InlineMath>: Rating of user v for item i
          </li>
          <li>
            <InlineMath>N(u)</InlineMath>: Set of users similar to u
          </li>
        </ul>
      </section>

      <section id="item-based-cf">
        <h2>Item-Based Collaborative Filtering</h2>
        <p>
          Item-based CF computes similarities between items based on user rating
          patterns and uses these to make predictions.
        </p>
        <h3>Mathematical Formulation:</h3>
        <BlockMath>
          {`
\\hat{r}_{ui} = \\frac{\\sum_{j \\in N(i)} sim(i,j) \\cdot r_{uj}}{\\sum_{j \\in N(i)} |sim(i,j)|}
          `}
        </BlockMath>
        <p>Where:</p>
        <ul>
          <li>
            <InlineMath>{`\hat{r}_{ui}`}</InlineMath>: Predicted rating of user
            u for item i
          </li>
          <li>
            <InlineMath>sim(i,j)</InlineMath>: Similarity between items i and j
          </li>
          <li>
            <InlineMath>{`r_{uj}`}</InlineMath>: Rating of user u for item j
          </li>
          <li>
            <InlineMath>N(i)</InlineMath>: Set of items similar to i
          </li>
        </ul>
      </section>

      <section id="matrix-factorization">
        <h2>Matrix Factorization</h2>
        <p>
          Matrix factorization is a latent factor model that represents users
          and items as vectors in a lower-dimensional space.
        </p>
        <h3>Mathematical Formulation:</h3>
        <BlockMath>{`R \\approx P \\cdot Q^T`}</BlockMath>
        <p>Where:</p>
        <ul>
          <li>
            <InlineMath>R</InlineMath>: User-item rating matrix
          </li>
          <li>
            <InlineMath>P</InlineMath>: User latent factor matrix
          </li>
          <li>
            <InlineMath>Q</InlineMath>: Item latent factor matrix
          </li>
        </ul>
        <p>Predicted rating:</p>
        <BlockMath>{`\\hat{r}_{ui} = p_u \\cdot q_i`}</BlockMath>
        <p>Optimization objective:</p>
        <BlockMath>
          {`\\min_{P,Q} \\sum_{(u,i) \\in K} (r_{ui} - p_u \\cdot q_i)^2 + \\lambda(\\|p_u\\|^2 + \\|q_i\\|^2)`}
        </BlockMath>
      </section>

      <section id="implementing-collaborative">
        <h2>Implementing Collaborative Filtering</h2>
        <p>
          Here's a basic implementation of matrix factorization using Python and
          NumPy:
        </p>
        <CodeBlock
          language="python"
          code={`
import numpy as np

class MatrixFactorization:
    def __init__(self, R, K, alpha=0.001, beta=0.02, iterations=100):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        for i in range(self.iterations):
            for u in range(self.num_users):
                for i in range(self.num_items):
                    if self.R[u][i] > 0:
                        eui = self.R[u][i] - np.dot(self.P[u,:], self.Q[i,:].T)
                        for k in range(self.K):
                            self.P[u][k] += self.alpha * (2 * eui * self.Q[i][k] - self.beta * self.P[u][k])
                            self.Q[i][k] += self.alpha * (2 * eui * self.P[u][k] - self.beta * self.Q[i][k])
            
    def predict(self, user, item):
        return np.dot(self.P[user,:], self.Q[item,:].T)

# Usage
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

mf = MatrixFactorization(R, K=2, alpha=0.1, beta=0.01, iterations=100)
mf.train()
print(mf.predict(user=1, item=2))  # Predict rating for user 1, item 2
          `}
        />
      </section>
    </Container>
  );
};

export default CollaborativeFiltering;
