import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const ExerciseGit = () => {
  const exerciseLinks = [
    {
      to: "/exercise1",
      label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
      component: lazy(() => import("pages/module1/exercise/Exercise1")),
    },
    {
      to: "/exercise2",
      label: "Exercise 2",
      component: lazy(() => import("pages/module1/exercise/Exercise2")),
    },
  ];

  const location = useLocation();
  const module = 1;
  return (
    <ModuleFrame
      module={1}
      isCourse={false}
      title="Module 1: Git Exercises"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Row>
            <p>
              Practice using Git for version control and GitHub for
              collaboration.
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

export default ExerciseGit;
