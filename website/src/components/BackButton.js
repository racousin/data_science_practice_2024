import React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowLeft } from "@fortawesome/free-solid-svg-icons";

const BackButton = () => {
  const navigate = useNavigate();

  const goBack = () => {
    navigate(-1); // Navigate back in the history stack
  };

  return (
    <Button variant="secondary" onClick={goBack} className="btn-back">
      <FontAwesomeIcon icon={faArrowLeft} /> Back
    </Button>
  );
};

export default BackButton;
