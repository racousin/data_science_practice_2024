import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseRecommendationSystems = () => {
  const courseLinks = []
  // const courseLinks = [
  //   {
  //     to: "/introduction",
  //     label: "Introduction to Recommendation Systems",
  //     component: lazy(() => import("pages/module12/course/Introduction")),
  //     subLinks: [
  //       {
  //         id: "what-are-recommendation-systems",
  //         label: "What are Recommendation Systems?",
  //       },
  //       {
  //         id: "types-of-recommendation-systems",
  //         label: "Types of Recommendation Systems",
  //       },
  //       { id: "applications", label: "Applications of Recommendation Systems" },
  //     ],
  //   },
  //   {
  //     to: "/content-based-filtering",
  //     label: "Content-Based Filtering",
  //     component: lazy(() =>
  //       import("pages/module12/course/ContentBasedFiltering")
  //     ),
  //     subLinks: [
  //       { id: "item-representation", label: "Item Representation" },
  //       { id: "user-profiles", label: "User Profiles" },
  //       { id: "similarity-measures", label: "Similarity Measures" },
  //       {
  //         id: "implementing-content-based",
  //         label: "Implementing Content-Based Filtering",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/collaborative-filtering",
  //     label: "Collaborative Filtering",
  //     component: lazy(() =>
  //       import("pages/module12/course/CollaborativeFiltering")
  //     ),
  //     subLinks: [
  //       { id: "user-based-cf", label: "User-Based Collaborative Filtering" },
  //       { id: "item-based-cf", label: "Item-Based Collaborative Filtering" },
  //       { id: "matrix-factorization", label: "Matrix Factorization" },
  //       {
  //         id: "implementing-collaborative",
  //         label: "Implementing Collaborative Filtering",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/hybrid-methods",
  //     label: "Hybrid Methods",
  //     component: lazy(() => import("pages/module12/course/HybridMethods")),
  //     subLinks: [
  //       { id: "weighted-hybrid", label: "Weighted Hybrid" },
  //       { id: "switching-hybrid", label: "Switching Hybrid" },
  //       { id: "feature-combination", label: "Feature Combination" },
  //       { id: "implementing-hybrid", label: "Implementing Hybrid Methods" },
  //     ],
  //   },
  //   {
  //     to: "/deep-learning-for-recommendations",
  //     label: "Deep Learning for Recommendations",
  //     component: lazy(() =>
  //       import("pages/module12/course/DeepLearningRecommendations")
  //     ),
  //     subLinks: [
  //       {
  //         id: "neural-collaborative-filtering",
  //         label: "Neural Collaborative Filtering",
  //       },
  //       {
  //         id: "autoencoders-for-recommendations",
  //         label: "Autoencoders for Recommendations",
  //       },
  //       {
  //         id: "sequence-models-for-recommendations",
  //         label: "Sequence Models for Recommendations",
  //       },
  //       {
  //         id: "implementing-deep-learning",
  //         label: "Implementing Deep Learning Models",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/context-aware-recommendations",
  //     label: "Context-Aware Recommendations",
  //     component: lazy(() =>
  //       import("pages/module12/course/ContextAwareRecommendations")
  //     ),
  //     subLinks: [
  //       { id: "contextual-information", label: "Contextual Information" },
  //       { id: "pre-filtering", label: "Pre-filtering" },
  //       { id: "post-filtering", label: "Post-filtering" },
  //       { id: "contextual-modeling", label: "Contextual Modeling" },
  //     ],
  //   },
  //   {
  //     to: "/evaluation-metrics",
  //     label: "Evaluation Metrics for Recommendation Systems",
  //     component: lazy(() => import("pages/module12/course/EvaluationMetrics")),
  //     subLinks: [
  //       { id: "accuracy-metrics", label: "Accuracy Metrics" },
  //       { id: "ranking-metrics", label: "Ranking Metrics" },
  //       { id: "diversity-and-novelty", label: "Diversity and Novelty" },
  //       { id: "user-studies", label: "User Studies" },
  //     ],
  //   },
  //   {
  //     to: "/cold-start-problem",
  //     label: "Cold Start Problem",
  //     component: lazy(() => import("pages/module12/course/ColdStartProblem")),
  //     subLinks: [
  //       { id: "new-user-problem", label: "New User Problem" },
  //       { id: "new-item-problem", label: "New Item Problem" },
  //       { id: "strategies-for-cold-start", label: "Strategies for Cold Start" },
  //     ],
  //   },
  //   {
  //     to: "/CaseStudy11",
  //     label: "CaseStudy",
  //     component: lazy(() => import("pages/module5/course/CaseStudy")),
  //   },
  // ];

  const location = useLocation();
  const module = 12;
  return (
    <ModuleFrame
      module={11}
      isCourse={true}
      title="Module 12: Recommendation Systems"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              This module covers the fundamentals and advanced topics in
              recommendation systems.
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

export default CourseRecommendationSystems;
