import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const ExerciseDataCollection = () => {
  const exerciseLinks = [
    // {
    //   to: "/exercise1",
    //   label: <>Exercise 1<span style={{color: 'red', fontWeight: 'bold'}}>*</span></>,
    //   component: lazy(() => import("pages/module4/exercise/Exercise1")),
    // },
    // Add links to other exercises as needed
  ];

  const location = useLocation();
  const module = 4;
  return (
    <ModuleFrame
      module={4}
      isCourse={false}
      title="Module 4: Data Collection Exercises"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Row>
            <p>
              In this module, you will practice retrieving data from different
              sources using Python.
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

export default ExerciseDataCollection;
