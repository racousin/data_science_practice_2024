import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const DeepLearningRecommendations = () => {
  return (
    <Container>
      <h1>Deep Learning for Recommendations</h1>

      <section id="introduction">
        <h2>Introduction</h2>
        <p>
          Deep learning models have shown significant promise in recommendation
          systems due to their ability to learn complex patterns and
          representations from large-scale data. These models can capture
          non-linear relationships and latent features that traditional methods
          might miss.
        </p>
      </section>

      <section id="neural-collaborative-filtering">
        <h2>Neural Collaborative Filtering (NCF)</h2>
        <p>
          NCF replaces the inner product in matrix factorization with a neural
          architecture to learn the user-item interaction function.
        </p>
        <h3>Mathematical Formulation:</h3>
        <BlockMath>{`\\hat{y}_{ui} = f(P^T v_u, Q^T v_i)`}</BlockMath>
        <p>Where:</p>
        <ul>
          <li>
            <InlineMath>{`\hat{y}_{ui}`}</InlineMath>: Predicted rating for user
            u and item i
          </li>
          <li>
            <InlineMath>P, Q</InlineMath>: User and item embedding matrices
          </li>
          <li>
            <InlineMath>v_u, v_i</InlineMath>: One-hot encoded vectors for user
            u and item i
          </li>
          <li>
            <InlineMath>f</InlineMath>: Neural network function
          </li>
        </ul>
      </section>

      <section id="autoencoders-for-recommendations">
        <h2>Autoencoders for Recommendations</h2>
        <p>
          Autoencoders can be used to learn compressed representations of
          user-item interactions and reconstruct the rating matrix.
        </p>
        <h3>Architecture:</h3>
        <CodeBlock
          language="python"
          code={`
Input layer: x (user's rating vector)
Encoder: h = σ(Wx + b)
Decoder: x̂ = σ(W'h + b')

Where:
σ: Activation function (e.g., ReLU)
W, W': Weight matrices
b, b': Bias vectors
          `}
        />
      </section>

      <section id="sequence-models-for-recommendations">
        <h2>Sequence Models for Recommendations</h2>
        <p>
          Recurrent Neural Networks (RNNs) and Transformers can model sequential
          user behavior for session-based recommendations.
        </p>
        <h3>RNN-based Model:</h3>
        <BlockMath>
          {`
h_t = \\tanh(W_h x_t + U_h h_{t-1} + b_h)
\\hat{y}_t = \\text{softmax}(W_y h_t + b_y)
          `}
        </BlockMath>
        <p>Where:</p>
        <ul>
          <li>
            <InlineMath>h_t</InlineMath>: Hidden state at time t
          </li>
          <li>
            <InlineMath>x_t</InlineMath>: Input at time t
          </li>
          <li>
            <InlineMath>W_h, U_h, W_y</InlineMath>: Weight matrices
          </li>
          <li>
            <InlineMath>b_h, b_y</InlineMath>: Bias vectors
          </li>
          <li>
            <InlineMath>{`\hat{y}_t`}</InlineMath>: Predicted next item
            probabilities
          </li>
        </ul>
      </section>

      <section id="implementing-deep-learning">
        <h2>Implementing Deep Learning Models</h2>
        <p>
          Here's a basic implementation of a Neural Collaborative Filtering
          model using PyTorch:
        </p>
        <CodeBlock
          language="python"
          code={`
import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, layers):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        self.output_layer = nn.Linear(layers[-1], 1)
        self.activation = nn.ReLU()

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        output = self.output_layer(x)
        return output.squeeze()

# Usage
num_users, num_items = 1000, 5000
model = NCF(num_users, num_items, embedding_size=64, layers=[128, 64, 32, 16])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop (simplified)
for epoch in range(num_epochs):
    for user, item, rating in dataloader:
        optimizer.zero_grad()
        prediction = model(user, item)
        loss = criterion(prediction, rating)
        loss.backward()
        optimizer.step()
          `}
        />
      </section>

      <section id="challenges">
        <h2>Challenges and Considerations</h2>
        <p>Deep learning models for recommendations face several challenges:</p>
        <ul>
          <li>Cold-start problem for new users/items</li>
          <li>Scalability to large user-item matrices</li>
          <li>Interpretability of model decisions</li>
          <li>Overfitting on sparse data</li>
          <li>Balancing exploration and exploitation</li>
        </ul>
      </section>
    </Container>
  );
};

export default DeepLearningRecommendations;
