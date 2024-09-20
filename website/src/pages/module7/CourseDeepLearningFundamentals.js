import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseDeepLearningFundamentals = () => {
  const courseLinks = []
  // const courseLinks = [
  //   {
  //     to: "/introduction",
  //     label: "Introduction to Neural Networks",
  //     component: lazy(() => import("pages/module7/course/Introduction")),
  //     subLinks: [
  //       {
  //         id: "biological-inspiration",
  //         label: "Biological inspiration and history",
  //       },
  //       {
  //         id: "basic-components",
  //         label: "Basic components: neurons, layers, activation functions",
  //       },
  //       { id: "feedforward", label: "Feedforward neural networks" },
  //     ],
  //   },
  //   {
  //     to: "/backpropagation",
  //     label: "Backpropagation and Optimization",
  //     component: lazy(() => import("pages/module7/course/Backpropagation")),
  //     subLinks: [
  //       { id: "chain-rule", label: "Chain rule and gradient descent" },
  //       { id: "backpropagation-algorithm", label: "Backpropagation algorithm" },
  //       {
  //         id: "sgd",
  //         label: "Stochastic Gradient Descent (SGD) and its variants",
  //       },
  //       {
  //         id: "optimizers",
  //         label: "Learning rate schedules and adaptive optimizers",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/autodiff",
  //     label: "Automatic Differentiation (Autodiff)",
  //     component: lazy(() => import("pages/module7/course/Autodiff")),
  //     subLinks: [
  //       { id: "forward-reverse", label: "Forward and reverse mode autodiff" },
  //       { id: "computational-graphs", label: "Computational graphs" },
  //       { id: "pytorch-autograd", label: "PyTorch's autograd system" },
  //     ],
  //   },
  //   {
  //     to: "/activation-functions",
  //     label: "Activation Functions",
  //     component: lazy(() => import("pages/module7/course/ActivationFunctions")),
  //     subLinks: [
  //       {
  //         id: "common-functions",
  //         label: "Sigmoid, tanh, ReLU, and their variants",
  //       },
  //       { id: "properties", label: "Properties and use cases" },
  //       {
  //         id: "custom-activations",
  //         label: "Implementing custom activation functions in PyTorch",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/regularization",
  //     label: "Regularization Techniques",
  //     component: lazy(() => import("pages/module7/course/Regularization")),
  //     subLinks: [
  //       { id: "l1-l2", label: "L1 and L2 regularization" },
  //       { id: "dropout", label: "Dropout" },
  //       { id: "batch-normalization", label: "Batch Normalization" },
  //       { id: "early-stopping", label: "Early stopping" },
  //     ],
  //   },
    // {
    //   to: "/cnn",
    //   label: "Convolutional Neural Networks (CNNs)",
    //   component: lazy(() => import("pages/module7/course/CNN")),
    //   subLinks: [
    //     { id: "convolution", label: "Convolution operation and intuition" },
    //     { id: "pooling", label: "Pooling layers" },
    //   ],
    // },
    // {
    //   to: "/rnn",
    //   label: "Recurrent Neural Networks (RNNs)",
    //   component: lazy(() => import("pages/module7/course/RNN")),
    //   subLinks: [
    //     {
    //       id: "sequence-modeling",
    //       label: "Sequence modeling and RNN architecture",
    //     },
    //     {
    //       id: "vanishing-exploding",
    //       label: "Vanishing and exploding gradients",
    //     },
    //     { id: "lstm", label: "Long Short-Term Memory (LSTM) networks" },
    //     { id: "gru", label: "Gated Recurrent Units (GRU)" },
    //   ],
    // },
    // {
    //   to: "/advanced-rnn",
    //   label: "Advanced RNN Architectures",
    //   component: lazy(() => import("pages/module7/course/AdvancedRNN")),
    //   subLinks: [
    //     { id: "bidirectional", label: "Bidirectional RNNs" },
    //     { id: "encoder-decoder", label: "Encoder-Decoder models" },
    //     { id: "attention", label: "Attention mechanisms" },
    //     { id: "transformer", label: "Transformer architecture" },
    //   ],
    // },
    // {
    //   to: "/training",
    //   label: "Training Deep Neural Networks",
    //   component: lazy(() => import("pages/module7/course/Training")),
    //   subLinks: [
    //     {
    //       id: "weight-initialization",
    //       label: "Weight initialization strategies",
    //     },
    //     { id: "gradient-clipping", label: "Gradient clipping" },
    //     { id: "curriculum-learning", label: "Curriculum learning" },
    //     { id: "mixed-precision", label: "Mixed precision training" },
    //   ],
    // },
    // {
    //   to: "/interpretation",
    //   label: "Model Interpretation and Visualization",
    //   component: lazy(() => import("pages/module7/course/Interpretation")),
    //   subLinks: [
    //     {
    //       id: "saliency-maps",
    //       label: "Saliency maps and activation maximization",
    //     },
    //     { id: "layer-visualization", label: "Layer visualization" },
    //     { id: "tsne", label: "t-SNE for high-dimensional data visualization" },
    //   ],
    // },
  //   {
  //     to: "/CaseStudy7",
  //     label: "CaseStudy",
  //     component: lazy(() => import("pages/module7/course/CaseStudy")),
  //   },
  // ];

  const location = useLocation();
  const module = 7;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 7: Deep Learning Fundamentals"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              This module covers the fundamentals of deep learning, from basic
              neural network concepts to advanced architectures and training
              techniques.
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

export default CourseDeepLearningFundamentals;
