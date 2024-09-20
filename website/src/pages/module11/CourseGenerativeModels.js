import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseGenerativeModels = () => {
  const courseLinks = [
    // {
    //   to: "/introduction",
    //   label: "Introduction to Generative Models",
    //   component: lazy(() => import("pages/module11/course/Introduction")),
    //   subLinks: [
    //     {
    //       id: "what-are-generative-models",
    //       label: "What are Generative Models?",
    //     },
    //     {
    //       id: "types-of-generative-models",
    //       label: "Types of Generative Models",
    //     },
    //     { id: "applications", label: "Applications of Generative Models" },
    //   ],
    // },
    // {
    //   to: "/autoencoders",
    //   label: "Autoencoders",
    //   component: lazy(() => import("pages/module11/course/Autoencoders")),
    //   subLinks: [
    //     { id: "basic-autoencoders", label: "Basic Autoencoders" },
    //     {
    //       id: "variational-autoencoders",
    //       label: "Variational Autoencoders (VAEs)",
    //     },
    //     {
    //       id: "applications-of-autoencoders",
    //       label: "Applications of Autoencoders",
    //     },
    //   ],
    // },
    // {
    //   to: "/gans",
    //   label: "Generative Adversarial Networks (GANs)",
    //   component: lazy(() => import("pages/module11/course/GANs")),
    //   subLinks: [
    //     { id: "gan-architecture", label: "GAN Architecture" },
    //     { id: "training-gans", label: "Training GANs" },
    //     { id: "gan-variants", label: "GAN Variants (DCGAN, WGAN, etc.)" },
    //     { id: "applications-of-gans", label: "Applications of GANs" },
    //   ],
    // },
    // {
    //   to: "/flow-based-models",
    //   label: "Flow-based Models",
    //   component: lazy(() => import("pages/module11/course/FlowBasedModels")),
    //   subLinks: [
    //     { id: "normalizing-flows", label: "Normalizing Flows" },
    //     { id: "real-nvp", label: "Real NVP" },
    //     { id: "glow", label: "Glow" },
    //   ],
    // },
    // {
    //   to: "/diffusion-models",
    //   label: "Diffusion Models",
    //   component: lazy(() => import("pages/module11/course/DiffusionModels")),
    //   subLinks: [
    //     {
    //       id: "ddpm",
    //       label: "Denoising Diffusion Probabilistic Models (DDPM)",
    //     },
    //     { id: "score-based-models", label: "Score-based Models" },
    //     {
    //       id: "applications-of-diffusion-models",
    //       label: "Applications of Diffusion Models",
    //     },
    //   ],
    // },
    // {
    //   to: "/energy-based-models",
    //   label: "Energy-based Models",
    //   component: lazy(() => import("pages/module11/course/EnergyBasedModels")),
    //   subLinks: [
    //     { id: "ebm-basics", label: "EBM Basics" },
    //     { id: "training-ebms", label: "Training Energy-based Models" },
    //     {
    //       id: "applications-of-ebms",
    //       label: "Applications of Energy-based Models",
    //     },
    //   ],
    // },
    // {
    //   to: "/transformer-based-generative-models",
    //   label: "Transformer-based Generative Models",
    //   component: lazy(() =>
    //     import("pages/module11/course/TransformerBasedModels")
    //   ),
    //   subLinks: [
    //     { id: "gpt-models", label: "GPT Models" },
    //     { id: "bert-for-generation", label: "BERT for Generation" },
    //     { id: "t5-and-bart", label: "T5 and BART" },
    //   ],
    // },
    // {
    //   to: "/evaluation-metrics",
    //   label: "Evaluation Metrics for Generative Models",
    //   component: lazy(() => import("pages/module11/course/EvaluationMetrics")),
    //   subLinks: [
    //     { id: "inception-score", label: "Inception Score" },
    //     { id: "fid", label: "Fréchet Inception Distance (FID)" },
    //     { id: "perplexity", label: "Perplexity" },
    //     { id: "human-evaluation", label: "Human Evaluation" },
    //   ],
    // },
    // {
    //   to: "/challenges-and-future-directions",
    //   label: "Challenges and Future Directions",
    //   component: lazy(() =>
    //     import("pages/module11/course/ChallengesAndFuture")
    //   ),
    //   subLinks: [
    //     { id: "mode-collapse", label: "Mode Collapse" },
    //     { id: "training-stability", label: "Training Stability" },
    //     { id: "ethical-considerations", label: "Ethical Considerations" },
    //     { id: "future-research", label: "Future Research Directions" },
    //   ],
    // },
  ];

  const location = useLocation();
  const module = 11;
  return (
    <ModuleFrame
      module={10}
      isCourse={true}
      title="Module 11: Generative Models"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
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

export default CourseGenerativeModels;
