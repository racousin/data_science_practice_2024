import React from "react";
import { Container } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const ContextAwareRecommendations = () => {
  return (
    <Container>
      <h1>Context-Aware Recommendations</h1>

      <section id="introduction">
        <h2>Introduction</h2>
        <p>
          Context-aware recommendation systems (CARS) incorporate contextual
          information into the recommendation process. This approach aims to
          provide more relevant recommendations by considering factors such as
          time, location, or user mood.
        </p>
      </section>

      <section id="contextual-information">
        <h2>Contextual Information</h2>
        <p>Contextual information can be categorized into different types:</p>
        <ul>
          <li>Temporal context (e.g., time of day, season)</li>
          <li>Spatial context (e.g., location, environment)</li>
          <li>Social context (e.g., alone, with friends)</li>
          <li>User state (e.g., mood, goal)</li>
          <li>Device context (e.g., mobile, desktop)</li>
        </ul>
      </section>

      <section id="pre-filtering">
        <h2>Pre-filtering</h2>
        <p>
          Pre-filtering approaches filter the data before applying traditional
          recommendation algorithms.
        </p>
        <CodeBlock
          code={`
def pre_filter(data, context):
    return data[data['context'] == context]

filtered_data = pre_filter(all_data, current_context)
recommendations = traditional_recommender(filtered_data)
          `}
          language="python"
        />
      </section>

      <section id="post-filtering">
        <h2>Post-filtering</h2>
        <p>
          Post-filtering approaches apply context-based filtering after
          generating recommendations.
        </p>
        <CodeBlock
          code={`
def post_filter(recommendations, context):
    return [rec for rec in recommendations if is_relevant(rec, context)]

initial_recommendations = traditional_recommender(all_data)
final_recommendations = post_filter(initial_recommendations, current_context)
          `}
          language="python"
        />
      </section>

      <section id="contextual-modeling">
        <h2>Contextual Modeling</h2>
        <p>
          Contextual modeling incorporates context directly into the
          recommendation algorithm.
        </p>
        <BlockMath>
          {`R: User \\times Item \\times Context \\rightarrow Rating`}
        </BlockMath>
        <p>A simple contextual modeling approach using matrix factorization:</p>
        <CodeBlock
          code={`
import numpy as np

class ContextualMF:
    def __init__(self, num_users, num_items, num_contexts, num_factors):
        self.user_factors = np.random.rand(num_users, num_factors)
        self.item_factors = np.random.rand(num_items, num_factors)
        self.context_factors = np.random.rand(num_contexts, num_factors)

    def predict(self, user, item, context):
        return np.dot(self.user_factors[user], self.item_factors[item]) * self.context_factors[context]

    def train(self, ratings, learning_rate=0.01, num_iterations=100):
        for _ in range(num_iterations):
            for user, item, context, rating in ratings:
                prediction = self.predict(user, item, context)
                error = rating - prediction
                
                self.user_factors[user] += learning_rate * (error * self.item_factors[item] * self.context_factors[context] - 0.01 * self.user_factors[user])
                self.item_factors[item] += learning_rate * (error * self.user_factors[user] * self.context_factors[context] - 0.01 * self.item_factors[item])
                self.context_factors[context] += learning_rate * (error * self.user_factors[user] * self.item_factors[item] - 0.01 * self.context_factors[context])

# Usage
model = ContextualMF(num_users=100, num_items=1000, num_contexts=5, num_factors=20)
ratings = [(0, 5, 2, 4.5), (1, 10, 1, 3.0), ...]  # (user, item, context, rating)
model.train(ratings)
          `}
          language="python"
        />
      </section>

      <section id="evaluation">
        <h2>Evaluation</h2>
        <p>
          Evaluating context-aware recommender systems often requires special
          considerations:
        </p>
        <ul>
          <li>Context-aware accuracy metrics (e.g., contextual NDCG)</li>
          <li>User studies to assess real-world effectiveness</li>
          <li>A/B testing in production environments</li>
        </ul>
        <p>Example of a context-aware evaluation metric:</p>
        <BlockMath>
          {`cNDCG@k = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{\\sum_{i=1}^k \\frac{2^{rel_i} - 1}{\\log_2(i+1)} \\cdot contextRelevance_i}{IDCG_k}`}
        </BlockMath>
        <p>
          Where <InlineMath>contextRelevance_i</InlineMath> is a measure of how
          relevant the item is in the given context.
        </p>
      </section>
    </Container>
  );
};

export default ContextAwareRecommendations;
