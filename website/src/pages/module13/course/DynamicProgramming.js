import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import CodeBlock from "components/CodeBlock";

const DynamicProgramming = () => {
  return (
    <Container fluid>
      <h2>Dynamic Programming in MDPs</h2>

      <Row className="mt-4">
        <Col>
          <h3>The MDP Solution</h3>
          <p>
            Dynamic Programming allows to resolve the MDP optimization problem{" "}
            <InlineMath math="\pi^* = \arg \max_{\pi} E_{\tau\sim \pi}[{G(\tau)}]" />
            . It is an iterative process:
          </p>
          <ul>
            <li>Policy initialization</li>
            <li>Policy evaluation</li>
            <li>Policy improvement</li>
          </ul>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Policy Evaluation</h3>
          <p>
            Policy Evaluation: compute the state-value{" "}
            <InlineMath math="V_\pi" /> for a given policy{" "}
            <InlineMath math="\pi" />. We initialize <InlineMath math="V_0" />{" "}
            arbitrarily. And we update it using:
          </p>
          <BlockMath
            math={`
            \\begin{aligned}
            V_{k+1}(s) &= \\mathbb{E}_\\pi [r + \\gamma V_k(s_{t+1}) | S_t = s]\\\\
            &= \\sum_a \\pi(a | s) \\sum_{s'} P(s' | s, a) (R(s,a) + \\gamma V_k(s'))
            \\end{aligned}
          `}
          />
          <p>
            <InlineMath math="V_\pi(s)" /> is a fixed point for this equation,
            so if <InlineMath math="(V_k)_{k\in \mathbb{N}}" /> converges, it
            converges to <InlineMath math="V_\pi" />.
          </p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Policy Improvement</h3>
          <p>
            Policy Improvement generates a better policy{" "}
            <InlineMath math="\pi' \geq \pi" /> by acting greedily. Compute{" "}
            <InlineMath math="Q" /> from <InlineMath math="V" /> (
            <InlineMath math="\forall a,s" />
            ):
          </p>
          <BlockMath
            math={`
            \\begin{aligned}
            Q_\\pi(s, a) &= \\mathbb{E} [R_{t+1} + \\gamma V_\\pi(S_{t+1}) | S_t=s, A_t=a]\\\\
            &= \\sum_{s'} P(s' | s, a) (R(s,a) + \\gamma V_\\pi(s'))
            \\end{aligned}
          `}
          />
          <p>
            Update greedily:{" "}
            <InlineMath math="\pi'(s) = \arg\max_{a \in \mathcal{A}} Q_\pi(s, a)" />{" "}
            (<InlineMath math="\forall s" />)
          </p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Policy Improvement Visualization</h3>
          <p>
            <InlineMath math="\pi' (s) = \arg\max_{a \in A} Q_{\pi}(s, a)" />
          </p>
          <Row>
            <Col md={4}>
              <Image
                src="/assets/module13/tikz_picture_16.png"
                alt="Initial Policy"
                fluid
              />
              <p className="text-center">Initial Policy (π)</p>
            </Col>
            <Col md={4}>
              <Image
                src="/assets/module13/tikz_picture_17.png"
                alt="Q-values"
                fluid
              />
              <p className="text-center">
                Q-values (Q<sub>π</sub>)
              </p>
            </Col>
            <Col md={4}>
              <Image
                src="/assets/module13/tikz_picture_18.png"
                alt="Improved Policy"
                fluid
              />
              <p className="text-center">Improved Policy (π')</p>
            </Col>
          </Row>
          <p className="text-center mt-3">
            Policy Improvement Process: Initial Policy → Q-values → Improved
            Policy
          </p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Dynamic Programming</h3>
          <p>
            Policy Iteration: iterative procedure to improve the policy when
            combining policy evaluation and improvement.
          </p>
          <BlockMath
            math={`
            \\pi_0 \\xrightarrow[]{\\text{evaluation}} V_{\\pi_0} \\xrightarrow[]{\\text{improve}}
            \\pi_1 \\xrightarrow[]{\\text{evaluation}}\\dots \\xrightarrow[]{\\text{improve}}
            \\pi_* \\xrightarrow[]{\\text{evaluation}} V_*
          `}
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Bellman Equations Optimality</h3>
          <p>Bellman equations for the optimal value functions:</p>
          <BlockMath
            math={`
            \\begin{aligned}
            V_*(s) &= \\max_{a \\in \\mathcal{A}} Q_*(s,a)\\\\
            Q_*(s, a) &= R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a V_*(s') \\\\
            V_*(s) &= \\max_{a \\in \\mathcal{A}} \\big( R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a V_*(s') \\big) \\\\
            Q_*(s, a) &= R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P_{ss'}^a \\max_{a' \\in \\mathcal{A}} Q_*(s', a')
            \\end{aligned}
          `}
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Take Home Message</h3>
          <p>
            Initialize <InlineMath math="\pi(s), \forall s" />
          </p>
          <ol>
            <li>
              Evaluate <InlineMath math="V_\pi (s), \forall s" /> (using{" "}
              <InlineMath math="\mathbb{P}^a_{ss'}" />)
            </li>
            <li>
              Compute <InlineMath math="Q_\pi(s,a), \forall s,a" /> (using{" "}
              <InlineMath math="\mathbb{P}^a_{ss'}" />)
            </li>
            <li>
              Update{" "}
              <InlineMath math="\pi'(s) = \max_a Q_\pi (s,a), \forall s" />
            </li>
            <li>
              While <InlineMath math="\pi'(s) \neq \pi(s)" /> do{" "}
              <InlineMath math="\pi(s) = \pi'(s)" /> and iterate
            </li>
          </ol>
          <p>
            Result: <InlineMath math="\pi = \arg \max_{\pi} E[G]" />
          </p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h2>Code: Dynamic Programming</h2>
          <CodeBlock
            code={`
# Example implementation of Policy Iteration
def policy_evaluation(policy, V, P, R, gamma, theta):
    # ... implementation ...

def policy_improvement(V, P, R, gamma):
    # ... implementation ...

def policy_iteration(P, R, gamma, theta):
    V = {s: 0 for s in P.keys()}
    policy = {s: np.random.choice(len(P[s])) for s in P.keys()}
    
    while True:
        V = policy_evaluation(policy, V, P, R, gamma, theta)
        new_policy = policy_improvement(V, P, R, gamma)
        
        if new_policy == policy:
            break
        policy = new_policy
    
    return policy, V
            `}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default DynamicProgramming;
