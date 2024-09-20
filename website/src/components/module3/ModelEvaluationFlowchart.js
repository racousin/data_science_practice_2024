import React from "react";
import { Card } from "react-bootstrap";
import { ArrowRight } from "react-bootstrap-icons";

const FlowchartStep = ({ text }) => (
  <Card className="mb-2">
    <Card.Body className="text-center">{text}</Card.Body>
  </Card>
);

const ModelEvaluationFlowchart = () => {
  return (
    <div className="d-flex flex-column align-items-center">
      <FlowchartStep text="Train on D_train" />
      <ArrowRight size={24} />
      <FlowchartStep text="Evaluate and tune on D_val" />
      <ArrowRight size={24} />
      <FlowchartStep text="Final assessment on D_test" />
    </div>
  );
};

export default ModelEvaluationFlowchart;
