import React from "react";
import { Card, ProgressBar } from "react-bootstrap";
import "styles/OverallProgress.css";

const OverallProgress = ({ progress }) => {
  return (
    <Card className="overall-progress  text-center mt-3 mb-4">
      <Card.Header as="h5">Overall Progress</Card.Header>
      <Card.Body>
        <Card.Title>{progress}% Complete</Card.Title>
        <ProgressBar now={progress} label={`${progress}%`} />
        <Card.Text className="mt-3">
          This is an overview of your current completion rate across all
          modules. Keep up the good work!
        </Card.Text>
      </Card.Body>
    </Card>
  );
};

export default OverallProgress;
