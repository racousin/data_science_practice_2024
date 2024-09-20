import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";

const MDP = () => {
  return (
    <Container fluid>
      <h2>Understanding Markov Decision Processes (MDPs)</h2>

      <Row className="justify-content-center">
        <Col xs={12} md={10} lg={8}>
          <div className="text-center">
            <Image
              src="/assets/module13/mdp.jpg"
              alt="MDP Illustration"
              fluid
            />
            <p>
              Illustrative example of an MDP, showcasing state transitions,
              actions, and rewards.
            </p>
          </div>
        </Col>
      </Row>

      <Row>
        <Col>
          <h3>First Glossary of MDPs</h3>
          <ul>
            <li>
              State Space (<InlineMath math="S" />)
            </li>
            <li>
              Action Space (<InlineMath math="A" />)
            </li>
            <li>
              Transition Model (<InlineMath math="P" />)
            </li>
            <li>
              Reward function (<InlineMath math="R" />)
            </li>
            <li>
              Policy (<InlineMath math="\pi" />)
            </li>
            <li>
              Trajectory (<InlineMath math="\tau" />)
            </li>
            <li>
              Return (<InlineMath math="G" />)
            </li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Simple Grid World Problem</h3>
          <p>
            Our environment is a 4x4 grid where an agent aims to reach a goal.
          </p>
          <div className="text-center">
            <Image
              src="/assets/module13/tikz_picture_1.png"
              alt="Grid World"
              fluid
            />
            <p>A: Agent, G: Goal</p>
          </div>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col md={6}>
          <h3 id="state-space">
            State Space (<InlineMath math="S" />)
          </h3>
          <p>16 discrete states.</p>
          <Image
            src="/assets/module13/tikz_picture_2.png"
            alt="State Space"
            fluid
          />
        </Col>
        <Col md={6}>
          <h3 id="action-space">
            Action Space (<InlineMath math="A" />)
          </h3>
          <p>4 discrete actions (Up, Down, Left, Right).</p>
          <Image
            src="/assets/module13/tikz_picture_3.png"
            alt="Action Space"
            fluid
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3 id="transition-model">
            Transition Model:{" "}
            <InlineMath math="P_{ss'}^a = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a]" />
          </h3>
          <Row>
            <Col md={6}>
              <p>Deterministic environment.</p>
              <Image
                src="/assets/module13/tikz_picture_4.png"
                alt="Deterministic Transition"
                fluid
              />
            </Col>
            <Col md={6}>
              <p>Stochastic environment.</p>
              <Image
                src="/assets/module13/tikz_picture_5.png"
                alt="Stochastic Transition"
                fluid
              />
            </Col>
          </Row>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3 id="reward-function">
            Reward function: <InlineMath math="r = R(s, a) = r(s')" />
          </h3>
          <Row>
            <Col md={6}>
              <p>Simple goal reward.</p>
              <Image
                src="/assets/module13/tikz_picture_6.png"
                alt="Simple Reward"
                fluid
              />
            </Col>
            <Col md={6}>
              <p>Other example of environment reward function.</p>
              <Image
                src="/assets/module13/tikz_picture_7.png"
                alt="Complex Reward"
                fluid
              />
            </Col>
          </Row>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3 id="policy">
            Policy: (<InlineMath math="\pi: S \rightarrow A" />)
          </h3>
          <p>
            Agent action in a state defined by its policy
            deterministic/stochastic
          </p>
          <Image src="/assets/module13/tikz_picture_8.png" alt="Policy" fluid />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>
            Trajectory:{" "}
            <InlineMath math="\tau_{\pi} = (s_0, a_0, s_1, a_1, ...)" />
          </h3>
          <Image
            src="/assets/module13/tikz_picture_9.png"
            alt="Trajectory"
            fluid
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>
            Return: <InlineMath math="G_t=\sum_{k=1}^T \gamma^k r_{t+k}" />
          </h3>
        </Col>
        <Row>
          <Col md={6}>
            <p>Cumulative rewards</p>
            <Image
              src="/assets/module13/tikz_picture_10.png"
              alt="Return"
              fluid
            />
          </Col>
          <Col md={6}>
            <p>Discounted rewards (0.95)</p>
            <Image
              src="/assets/module13/tikz_picture_11.png"
              alt="Optimal Policy"
              fluid
            />
          </Col>
        </Row>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Objective: Find best Policy</h3>
          <BlockMath math="\pi^* = \arg \max_{\pi} E_{\tau\sim \pi}[{G(\tau)}]" />
          <p>Optimal policy in the grid world environment.</p>
          <Image
            src="/assets/module13/tikz_picture_12.png"
            alt="Optimal Policy"
            fluid
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h2>Code: Environment and Agent Interaction</h2>
          {/* Add coding examples or interactive elements here */}
          <CodeBlock
            code={`# Example code for environment and agent interaction
class GridWorld:
    # ... (environment implementation)

class Agent:
    # ... (agent implementation)

env = GridWorld()
agent = Agent()

for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        if done:
            break`}
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Second Glossary of MDPs</h3>
          <ul>
            <li>
              <a href="#value-function">
                Value Function (<InlineMath math="V" />)
              </a>
            </li>
            <li>
              <a href="#value-function">
                Action Value Function (<InlineMath math="Q" />)
              </a>
            </li>
            <li>
              <a href="#bellman-equations">Bellman Equations</a>
            </li>
            <li>Dynamic Programming</li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3 id="value-function">
            Value Function:{" "}
            <InlineMath math="V^{\pi}(s) = E_{\tau \sim \pi}[{G_t\left| S_t = s\right.}]" />
          </h3>
          <p>
            Expected Return for State following <InlineMath math="\pi" />
          </p>
          <Image
            src="/assets/module13/tikz_picture_13.png"
            alt="Value Function"
            fluid
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>
            Action Value Function:{" "}
            <InlineMath math="Q^{\pi}(s,a) = E_{\tau \sim \pi}[{G_t\left| S_t = s, A_t = a\right.}]" />
          </h3>
          <p>
            Expected Return for State-Action following <InlineMath math="\pi" />
          </p>
          <Image
            src="/assets/module13/tikz_picture_14.png"
            alt="Action Value Function"
            fluid
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3 id="bellman-equations">Bellman Equations</h3>
          <p>
            <strong>Idea:</strong> The value of your starting point is the
            reward you expect to get from being there, plus the value of
            wherever you land next.
          </p>
          <BlockMath
            math={`
      \\begin{aligned}
      V(s) &= \\mathbb{E}[G_t \\vert S_t = s] \\\\
      &= \\mathbb{E} [R_{t+1} + \\gamma G_{t+1} \\vert S_t = s] \\\\
      &= \\mathbb{E} [R_{t+1} + \\gamma V(S_{t+1}) \\vert S_t = s]
      \\end{aligned}
    `}
          />
          <BlockMath math="Q(s, a) = \mathbb{E} [R_{t+1} + \gamma \mathbb{E}_{a\sim\pi} Q(S_{t+1}, a) \mid S_t = s, A_t = a]" />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>
            Value Function Decomposition: <InlineMath math="V^{\pi}(s)" />
          </h3>
          <p className="text-center">
            <strong>Value Function:</strong>{" "}
            <InlineMath math="V^{\pi}(s) = E[R_{t+1} + \gamma V^{\pi}(S_{t+1})|S_t = s]" />
          </p>
          <div className="text-center">
            <Image
              src="/assets/module13/tikz_picture_15.png"
              alt="Value Function Decomposition"
              fluid
            />
          </div>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Bellman Equations Development</h3>
          <BlockMath
            math={`
      \\begin{aligned}
      V_{\\pi}(s) &= \\sum_{a \\in \\mathcal{A}} \\pi(a \\vert s) Q_{\\pi}(s, a) \\\\
      Q_{\\pi}(s, a) &= R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a V_{\\pi} (s') \\\\
      V_{\\pi}(s) &= \\sum_{a \\in \\mathcal{A}} \\pi(a \\vert s) \\big( R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a V_{\\pi} (s') \\big) \\\\
      Q_{\\pi}(s, a) &= R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a \\sum_{a' \\in \\mathcal{A}} \\pi(a' \\vert s') Q_{\\pi} (s', a')
      \\end{aligned}
    `}
          />
        </Col>
      </Row>

      {/* Continue with the rest of the content... */}
    </Container>
  );
};

export default MDP;
