import React from "react";
import { Container, Row, Col, Table } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const RL_frameworks = () => {
  return (
    <Container fluid>
      <h2>Reinforcement Learning Frameworks and Libraries</h2>

      <Row className="mt-4">
        <Col>
          <p>
            Reinforcement Learning (RL) has seen significant growth in recent
            years, leading to the development of various frameworks and
            libraries. These tools make it easier for researchers and
            practitioners to implement and experiment with RL algorithms. Here's
            an overview of some popular RL frameworks:
          </p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>1. OpenAI Gym</h3>
          <p>
            OpenAI Gym is a toolkit for developing and comparing reinforcement
            learning algorithms. It provides a wide variety of environments and
            a standardized API.
          </p>
          <CodeBlock
            code={`
import gym

env = gym.make('CartPole-v1')
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()

env.close()
            `}
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>2. Stable Baselines3</h3>
          <p>
            Stable Baselines3 is a set of improved implementations of RL
            algorithms based on OpenAI Baselines. It's compatible with OpenAI
            Gym environments and provides a simple interface for training
            agents.
          </p>
          <CodeBlock
            code={`
from stable_baselines3 import PPO
import gym

env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
            `}
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>3. RLlib</h3>
          <p>
            RLlib is an open-source library for reinforcement learning that
            offers both high scalability and a unified API for a variety of
            applications. It's part of the Ray project and supports TensorFlow,
            PyTorch, and JAX.
          </p>
          <CodeBlock
            code={`
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

ray.init()
config = {
    "env": "CartPole-v1",
    "num_workers": 2,
    "framework": "torch"
}

stop = {
    "training_iteration": 50,
    "timesteps_total": 100000,
    "episode_reward_mean": 195
}

results = tune.run(PPOTrainer, config=config, stop=stop)
            `}
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>4. Dopamine</h3>
          <p>
            Dopamine is a research framework for fast prototyping of
            reinforcement learning algorithms. It aims to fill the need for a
            small, easily grokked codebase in which users can freely experiment
            with wild ideas (speculative research).
          </p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>5. KerasRL</h3>
          <p>
            KerasRL is a high-level reinforcement learning library that is
            compatible with OpenAI Gym and implements state-of-the-art Deep RL
            algorithms. It's built on top of Keras, making it easy to use and
            extend.
          </p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <h3>Comparison of RL Frameworks</h3>
          <Table striped bordered hover>
            <thead>
              <tr>
                <th>Framework</th>
                <th>Ease of Use</th>
                <th>Flexibility</th>
                <th>Performance</th>
                <th>Community Support</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>OpenAI Gym</td>
                <td>High</td>
                <td>Medium</td>
                <td>Medium</td>
                <td>High</td>
              </tr>
              <tr>
                <td>Stable Baselines3</td>
                <td>High</td>
                <td>Medium</td>
                <td>High</td>
                <td>Medium</td>
              </tr>
              <tr>
                <td>RLlib</td>
                <td>Medium</td>
                <td>High</td>
                <td>High</td>
                <td>Medium</td>
              </tr>
              <tr>
                <td>Dopamine</td>
                <td>Medium</td>
                <td>High</td>
                <td>High</td>
                <td>Low</td>
              </tr>
              <tr>
                <td>KerasRL</td>
                <td>High</td>
                <td>Medium</td>
                <td>Medium</td>
                <td>Medium</td>
              </tr>
            </tbody>
          </Table>
        </Col>
      </Row>
    </Container>
  );
};

export default RL_frameworks;
