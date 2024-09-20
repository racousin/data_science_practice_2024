import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseReinforcementLearning = () => {
  const courseLinks = []
  // const courseLinks = [
  //   {
  //     to: "/mdp",
  //     label: "Markov Decision Processes",
  //     component: lazy(() => import("pages/module13/course/MDP")),
  //     subLinks: [
  //       { id: "state-space", label: "State Space" },
  //       { id: "action-space", label: "Action Space" },
  //       { id: "transition-model", label: "Transition Model" },
  //       { id: "reward-function", label: "Reward Function" },
  //       { id: "policy", label: "Policy" },
  //       { id: "value-function", label: "Value Function" },
  //       { id: "bellman-equations", label: "Bellman Equations" },
  //     ],
  //   },
  //   {
  //     to: "/dynamic-programming",
  //     label: "Dynamic Programming",
  //     component: lazy(() => import("pages/module13/course/DynamicProgramming")),
  //     subLinks: [
  //       { id: "policy-evaluation", label: "Policy Evaluation" },
  //       { id: "policy-improvement", label: "Policy Improvement" },
  //       { id: "policy-iteration", label: "Policy Iteration" },
  //       { id: "value-iteration", label: "Value Iteration" },
  //     ],
  //   },
  //   {
  //     to: "/model-free-methods",
  //     label: "Model-Free Methods",
  //     component: lazy(() => import("pages/module13/course/ModelFreeMethods")),
  //     subLinks: [
  //       { id: "monte-carlo", label: "Monte Carlo Methods" },
  //       { id: "td-learning", label: "Temporal Difference Learning" },
  //       { id: "sarsa", label: "SARSA" },
  //       { id: "q-learning", label: "Q-Learning" },
  //     ],
  //   },
  //   {
  //     to: "/deep-rl",
  //     label: "Deep Reinforcement Learning",
  //     component: lazy(() => import("pages/module13/course/DeepRL")),
  //     subLinks: [
  //       { id: "deep-q-learning", label: "Deep Q-Learning" },
  //       { id: "policy-gradient", label: "Policy Gradient Methods" },
  //       { id: "actor-critic", label: "Actor-Critic Methods" },
  //     ],
  //   },
  //   {
  //     to: "/exploration-exploitation",
  //     label: "Exploration vs. Exploitation",
  //     component: lazy(() =>
  //       import("pages/module13/course/ExplorationExploitation")
  //     ),
  //     subLinks: [
  //       { id: "epsilon-greedy", label: "Epsilon-Greedy Strategy" },
  //       { id: "ucb", label: "Upper Confidence Bound" },
  //       { id: "thompson-sampling", label: "Thompson Sampling" },
  //     ],
  //   },
  //   {
  //     to: "/advanced-topics",
  //     label: "Advanced Topics",
  //     component: lazy(() => import("pages/module13/course/AdvancedTopics")),
  //     subLinks: [
  //       { id: "multi-agent-rl", label: "Multi-Agent RL" },
  //       { id: "inverse-rl", label: "Inverse Reinforcement Learning" },
  //       { id: "hierarchical-rl", label: "Hierarchical RL" },
  //     ],
  //   },
  //   {
  //     to: "/RL frameworks",
  //     label: "RL frameworks",
  //     component: lazy(() => import("pages/module13/course/RL_frameworks")),
  //     subLinks: [],
  //   },
  //   {
  //     to: "/case-study",
  //     label: "Case Study",
  //     component: lazy(() => import("pages/module13/course/CaseStudy")),
  //   },
  // ];

  const location = useLocation();
  const module = 13;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 13: Reinforcement Learning"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about reinforcement learning, its
              fundamental concepts, algorithms, and applications in artificial
              intelligence and decision-making processes.
            </p>
          </Row>
          <Row>
            <Col>
              <p>Last Updated: {"2024-09-20"}</p>
            </Col>
          </Row>
        </>
      )}
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseReinforcementLearning;
