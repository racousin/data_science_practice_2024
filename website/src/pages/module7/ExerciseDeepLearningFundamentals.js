import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const ExerciseDeepLearningFundamentals = () => {
  const exerciseLinks = [
    // {
    //   to: "/exercise1",
    //   label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
    //   component: lazy(() => import("pages/module7/exercise/Exercise1")),
    // },
    // Add links to other exercises as needed
  ];

  const location = useLocation();
  const module = 7;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 8: Exercise Deep Learning Fundamentals"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Row>
            <p>
              In this module, you will practice building and training neural
              networks using PyTorch.
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
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default ExerciseDeepLearningFundamentals;
