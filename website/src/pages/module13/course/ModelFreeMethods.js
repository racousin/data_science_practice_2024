import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";

const ModelFreeMethods = () => {
  return (
    <Container fluid>
      <h2>Model-Free Reinforcement Learning Methods</h2>

      <Row className="mt-4">
        <Col>
          <h3>Reinforcement Learning Objective</h3>
          <p>
            RL aims to optimize decision-making in environments without a known
            transition model <InlineMath math="P" />.
          </p>
          <BlockMath
            math={`
              \\begin{align*}
              \\pi^* &= \\arg\\max_{\\pi} J(\\pi)\\\\
              J(\\pi) &= \\mathbb{E}_{\\tau\\sim\\pi}[G(\\tau)] = \\int_{\\tau} \\mathbb{P}(\\tau|\\pi) G(\\tau)
              \\end{align*}
            `}
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>First Glossary of RL</h3>
          <ul>
            <li>Model free / Model based</li>
            <li>Q-learning / Policy Optimization</li>
            <li>On-policy / Off-policy</li>
            <li>
              <InlineMath math="\epsilon" />
              -Greedy
            </li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Overview of RL Algorithms</h3>
          <Image
            src="/assets/module13/tikz_images_2/tax.png"
            alt="RL Algorithms Taxonomy"
            fluid
          />
          <ul>
            <li>
              <strong>Model free:</strong> learn the policy{" "}
              <InlineMath math="\pi^*" /> directly
            </li>
            <li>
              <strong>Model based:</strong> use an environment model{" "}
              <InlineMath math="P^*" /> to learn <InlineMath math="\pi^*" />
            </li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Key Strategies in Model-Free RL</h3>
          <ul>
            <li>
              <strong>Q-learning:</strong> Learn the action-value function{" "}
              <InlineMath math="Q" /> to determine the best action given a
              state:
              <BlockMath math="\pi(s) = \arg\max_{a \in \mathcal{A}} Q_\pi(s, a)" />
            </li>
            <li>
              <strong>Policy Optimization:</strong> Directly learn the policy{" "}
              <InlineMath math="\pi" /> that maximizes the expected return.
            </li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Exploration-Exploitation</h3>
          <p>
            Knowledge of the environment comes from interaction. There are
            trade-offs to be made between using what we know and further
            exploration.
          </p>
          <Image
            src="/assets/module13/tikz_images_2/explore_vs_exploit.jpeg"
            alt="Explore vs Exploit"
            fluid
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>
            The <InlineMath math="\epsilon" />
            -Greedy Strategy
          </h3>
          <p>
            The <InlineMath math="\epsilon" />
            -greedy strategy is a simple yet effective method for balancing
            exploration and exploitation by choosing:
          </p>
          <ul>
            <li>
              With probability <InlineMath math="\epsilon" />, choose an action
              at random (exploration).
            </li>
            <li>
              With probability <InlineMath math="1 - \epsilon" />, choose the
              action with the highest estimated value (exploitation).
            </li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>On-policy vs Off-policy</h3>
          <ul>
            <li>
              <strong>On-policy:</strong> Directly learns from and improves the
              policy it executes.
            </li>
            <li>
              <strong>Off-policy:</strong> Learns a different policy from the
              executed one, allowing for learning from observations.
            </li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Model-Free RL Methods</h3>
          <ul>
            <li>Monte-Carlo</li>
            <li>Temporal difference</li>
            <li>SARSA</li>
            <li>Q-learning</li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Monte-Carlo Method</h3>
          <p>
            To evaluate{" "}
            <InlineMath math="V_\pi(s) = E_{\tau \sim \pi}[{G_t\left| s_t = s\right.}]" />
            :
          </p>
          <ol>
            <li>
              Generate an episode with the policy <InlineMath math="\pi" />
            </li>
            <li>
              Compute{" "}
              <InlineMath math="G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}" />
            </li>
            <li>
              Update the value function:
              <BlockMath math="V_\pi(s) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s]}" />
            </li>
          </ol>
          <p>Similarly, for the action-value function:</p>
          <BlockMath math="Q_\pi(s, a) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}" />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Monte-Carlo Algorithm</h3>
          <ol>
            <li>
              Generate an episode with the policy <InlineMath math="\pi" />{" "}
              (extract from <InlineMath math="Q" />{" "}
              <InlineMath math="\epsilon" />
              -greedy)
            </li>
            <li>
              Update Q using the episode:
              <BlockMath math="q_\pi(s, a) = \frac{\sum_{t=1}^T \big( \mathbb{1}[S_t = s, A_t = a] \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \big)}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]}" />
            </li>
            <li>Iterate</li>
          </ol>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Visual Steps in Monte Carlo</h3>
          <Row>
            <Col md={4}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_1.png"
                alt="Generate Episode"
                fluid
              />
              <p className="text-center">
                1. Generate episode following{" "}
                <InlineMath math="\arg\max Q(s, a)" />
              </p>
            </Col>
            <Col md={4}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_2.png"
                alt="Evaluate Q"
                fluid
              />
              <p className="text-center">2. Evaluate Q</p>
            </Col>
            <Col md={4}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_3.png"
                alt="Iterate"
                fluid
              />
              <p className="text-center">3. Iterate</p>
            </Col>
          </Row>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Temporal Difference (TD) Learning</h3>
          <p>
            TD Learning combines Monte Carlo and dynamic programming ideas,
            using bootstrapping for value updates.
          </p>
          <h4>Bellman equations:</h4>
          <BlockMath math="V(S_t) = \mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s]" />
          <BlockMath math="Q(s, a) = \mathbb{E} [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]" />
          <h4>TD Target (unbiased estimate):</h4>
          <ul>
            <li>
              For <InlineMath math="V(S_t)" />:{" "}
              <InlineMath math="R_{t+1} + \gamma V(S_{t+1})" />
            </li>
            <li>
              For <InlineMath math="Q(S_t, A_t)" />:{" "}
              <InlineMath math="R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})" />
            </li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>TD Learning - Value Function Estimation</h3>
          <h4>
            TD Error (<InlineMath math="\delta_t" />
            ):
          </h4>
          <BlockMath math="\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)" />
          <h4>Update Rule:</h4>
          <BlockMath math="V(S_t) \leftarrow V(S_t) + \alpha \delta_t" />
          <p>
            Where <InlineMath math="\alpha" /> is the learning rate.
          </p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>SARSA Algorithm</h3>
          <ol>
            <li>
              Initialize <InlineMath math="Q(s,a)" /> for all{" "}
              <InlineMath math="s,a" />
            </li>
            <li>
              For each episode:
              <ul>
                <li>
                  Initialize <InlineMath math="S_t" />
                </li>
                <li>
                  Choose <InlineMath math="A_t" /> from{" "}
                  <InlineMath math="S_t" /> using policy derived from{" "}
                  <InlineMath math="Q" /> (e.g., <InlineMath math="\epsilon" />
                  -greedy)
                </li>
                <li>
                  For each step of episode:
                  <ul>
                    <li>
                      Take action <InlineMath math="A_t" />, observe{" "}
                      <InlineMath math="R_{t+1}" /> and{" "}
                      <InlineMath math="S_{t+1}" />
                    </li>
                    <li>
                      Choose <InlineMath math="A_{t+1}" /> from{" "}
                      <InlineMath math="S_{t+1}" /> using policy derived from{" "}
                      <InlineMath math="Q" />
                    </li>
                    <li>
                      Update <InlineMath math="Q" />:
                      <BlockMath math="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]" />
                    </li>
                    <li>
                      <InlineMath math="S_t \leftarrow S_{t+1}; A_t \leftarrow A_{t+1}" />
                    </li>
                  </ul>
                </li>
              </ul>
            </li>
          </ol>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Visual Steps in SARSA</h3>
          <Row>
            <Col md={3}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_4.png"
                alt="SARSA Step 1"
                fluid
              />
              <p className="text-center">
                1. Choose action <InlineMath math="a_3=\arg \max Q(s_3,a)" />
              </p>
            </Col>
            <Col md={3}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_5.png"
                alt="SARSA Step 2"
                fluid
              />
              <p className="text-center">
                2. Update <InlineMath math="Q(s_2, a_2)" /> with{" "}
                <InlineMath math="r_3 + \gamma Q(s_3, a_3)" />
              </p>
            </Col>
            <Col md={3}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_6.png"
                alt="SARSA Step 3"
                fluid
              />
              <p className="text-center">
                3. Choose action <InlineMath math="a_4=\arg \max Q(s_4,a)" />
              </p>
            </Col>
            <Col md={3}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_7.png"
                alt="SARSA Step 4"
                fluid
              />
              <p className="text-center">
                4. Update <InlineMath math="Q(s_3, a_3)" /> with{" "}
                <InlineMath math="r_4 + \gamma Q(s_4, a_4)" />
              </p>
            </Col>
          </Row>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Q-learning Algorithm</h3>
          <ol>
            <li>
              Initialize <InlineMath math="Q(s,a)" /> for all{" "}
              <InlineMath math="s,a" />
            </li>
            <li>
              For each episode:
              <ul>
                <li>
                  Initialize <InlineMath math="S_t" />
                </li>
                <li>
                  For each step of episode:
                  <ul>
                    <li>
                      Choose <InlineMath math="A_t" /> from{" "}
                      <InlineMath math="S_t" /> using policy derived from{" "}
                      <InlineMath math="Q" /> (e.g.,{" "}
                      <InlineMath math="\epsilon" />
                      -greedy)
                    </li>
                    <li>
                      Take action <InlineMath math="A_t" />, observe{" "}
                      <InlineMath math="R_{t+1}" /> and{" "}
                      <InlineMath math="S_{t+1}" />
                    </li>
                    <li>
                      Update <InlineMath math="Q" />:
                      <BlockMath math="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]" />
                    </li>
                    <li>
                      <InlineMath math="S_t \leftarrow S_{t+1}" />
                    </li>
                  </ul>
                </li>
              </ul>
            </li>
          </ol>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Visual Steps in Q-Learning</h3>
          <Row>
            <Col md={3}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_8.png"
                alt="Q-Learning Step 1"
                fluid
              />
              <p className="text-center">
                1. Choose action <InlineMath math="a_3=\arg \max Q(s_3,a)" />
              </p>
            </Col>
            <Col md={3}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_9.png"
                alt="Q-Learning Step 2"
                fluid
              />
              <p className="text-center">
                2. Update <InlineMath math="Q(s_3, a_3)" /> with{" "}
                <InlineMath math="r_4 + \gamma \max Q(s_4, a)" />
              </p>
            </Col>
            <Col md={3}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_10.png"
                alt="Q-Learning Step 3"
                fluid
              />
              <p className="text-center">
                3. Choose action <InlineMath math="a_4=\arg \max Q(s_4,a)" />
              </p>
            </Col>
            <Col md={3}>
              <Image
                src="/assets/module13/tikz_images_2/tikz_picture_11.png"
                alt="Q-Learning Step 4"
                fluid
              />
              <p className="text-center">
                4. Update <InlineMath math="Q(s_4, a_4)" /> with{" "}
                <InlineMath math="r_5 + \gamma \max Q(s_4, a)" />
              </p>
            </Col>
          </Row>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Code: Temporal Difference</h2>
          <CodeBlock
            code={`
# Example implementation of Q-learning
import numpy as np

def q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state, :])  # Exploit
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            
            state = next_state
    
    return Q

# Usage
# Q = q_learning(env, num_episodes=1000)
            `}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ModelFreeMethods;
