// StartLearningButton.js
import React from "react";
import { Link } from "react-router-dom";
import "styles/StartLearningButton.css"; // Assuming the CSS is stored here

const StartLearningButton = () => {
  return (
    <Link to="/teaching" className="start-learning-button">
      Start Learning
      <span className="button-overlay"></span>
    </Link>
  );
};

export default StartLearningButton;
