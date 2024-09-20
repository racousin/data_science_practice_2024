import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";

const CaseStudy = () => {
  const trainDataUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_course_train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_course_test.csv";
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_requirements.txt";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_course.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module4/course/module4_course.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module4/course/module4_course.ipynb";
  const metadata = {
    description:
      "This dataset contains information about passengers aboard the Titanic, including various demographic and travel details, as well as their survival status.",
    source: "Titanic Passenger Records",
    target: "survival",
    listData: [
      {
        name: "survival",
        description: "Survival status (0 = No, 1 = Yes)",
      },
      {
        name: "class",
        description: "Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)",
      },
      {
        name: "sex",
        description: "Sex of the passenger (0 = female, 1 = male)",
      },
      {
        name: "age",
        description: "Age of the passenger",
      },
      {
        name: "sibsp",
        description: "Number of Siblings/Spouses Aboard",
      },
      {
        name: "parch",
        description: "Number of Parents/Children Aboard",
      },
      {
        name: "fare",
        description: "Passenger Fare",
      },
      {
        name: "cabin",
        description: "Have Cabin (0 = No, 1 = Yes)",
      },
      {
        name: "embarked",
        description: "Port of Embarkation (encoded as integers)",
      },
    ],
  };
  return (
    <Container fluid>
      <h1 className="my-4">EDA and Model Baseline Case Study</h1>
      <DataInteractionPanel
        trainDataUrl={trainDataUrl}
        testDataUrl={testDataUrl}
        notebookUrl={notebookUrl}
        notebookHtmlUrl={notebookHtmlUrl}
        notebookColabUrl={notebookColabUrl}
        requirementsUrl={requirementsUrl}
      />
    </Container>
  );
};

export default CaseStudy;
