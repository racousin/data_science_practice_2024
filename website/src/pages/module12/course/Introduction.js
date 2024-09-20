import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { BlockMath, InlineMath } from "react-katex";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Recommendation Systems</h1>

      <section id="what-are-recommendation-systems">
        <h2>What are Recommendation Systems?</h2>
        <p>
          Recommendation systems are sophisticated algorithms designed to
          predict user preferences and suggest relevant items or content.
          Mathematically, we can represent a recommendation system as a
          function:
        </p>
        <BlockMath math="f: U \times I \rightarrow R" />
        <p>
          Where:
          <ul>
            <li>
              <InlineMath math="U" /> is the set of users
            </li>
            <li>
              <InlineMath math="I" /> is the set of items
            </li>
            <li>
              <InlineMath math="R" /> is the set of ratings or preferences
            </li>
          </ul>
        </p>
      </section>

      <section id="types-of-recommendation-systems">
        <h2>Types of Recommendation Systems</h2>
        <ol>
          <li>
            <strong>Content-Based Filtering:</strong> Recommends items similar
            to those a user has liked in the past.
            <BlockMath math="\text{similarity}(i_1, i_2) = \cos(\vec{v_{i_1}}, \vec{v_{i_2}}) = \frac{\vec{v_{i_1}} \cdot \vec{v_{i_2}}}{|\vec{v_{i_1}}| |\vec{v_{i_2}}|}" />
          </li>
          <li>
            <strong>Collaborative Filtering:</strong> Recommends items based on
            the preferences of similar users.
            <BlockMath math="r_{ui} = \bar{r_u} + \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r_v})}{\sum_{v \in N(u)} |\text{sim}(u, v)|}" />
          </li>
          <li>
            <strong>Hybrid Methods:</strong> Combines multiple recommendation
            techniques.
          </li>
        </ol>
      </section>

      <section id="applications">
        <h2>Applications of Recommendation Systems</h2>
        <ul>
          <li>E-commerce product recommendations</li>
          <li>Streaming service content suggestions</li>
          <li>Social media friend and content recommendations</li>
          <li>News article personalization</li>
          <li>Job and career recommendations</li>
        </ul>
      </section>

      <section id="basic-implementation">
        <h2>Basic Implementation Example</h2>
        <p>
          Here's a simple content-based recommendation system using cosine
          similarity:
        </p>
        <CodeBlock
          language="python"
          code={`
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Item features (rows are items, columns are features)
item_features = np.array([
    [1, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 1, 0]
])

# User profile
user_profile = np.array([1, 1, 0, 1])

# Calculate cosine similarity
similarities = cosine_similarity(user_profile.reshape(1, -1), item_features)

# Get recommendations (indices of items sorted by similarity)
recommendations = similarities.argsort()[0][::-1]

print("Recommended items (by index):", recommendations)
          `}
        />
      </section>

      <section id="challenges">
        <h2>Challenges in Recommendation Systems</h2>
        <ul>
          <li>
            <strong>Cold Start Problem:</strong> Difficulty in recommending to
            new users or items.
          </li>
          <li>
            <strong>Scalability:</strong> Handling large-scale datasets
            efficiently.
          </li>
          <li>
            <strong>Sparsity:</strong> Dealing with sparse user-item interaction
            matrices.
          </li>
          <li>
            <strong>Privacy Concerns:</strong> Balancing personalization with
            user privacy.
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default Introduction;
