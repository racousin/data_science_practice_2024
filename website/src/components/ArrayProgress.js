import React from "react";
import { ProgressBar } from "react-bootstrap";

const ArrayProgress = ({ progressPercent }) => {
  const getProgressVariant = () => {
    if (progressPercent < 33) return "danger"; // Red for low progress
    if (progressPercent < 66) return "warning"; // Yellow for medium progress
    return "success"; // Green for high progress
  };

  return (
    <ProgressBar
      now={progressPercent}
      label={`${progressPercent.toFixed(0)}%`}
      variant={getProgressVariant()}
    />
  );
};

export default ArrayProgress;
