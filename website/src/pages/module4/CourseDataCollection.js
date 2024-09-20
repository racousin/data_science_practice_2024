import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseDataCollection = () => {
  const courseLinks = []
  // const courseLinks = [
  //   {
  //     to: "/introduction",
  //     label: "Introduction to Data Collection",
  //     component: lazy(() => import("pages/module4/course/Introduction")),
  //     subLinks: [
  //       { id: "importance", label: "Importance in Data Science Pipeline" },
  //       { id: "types", label: "Types of Data" },
  //       { id: "sources", label: "Data Sources" },
  //     ],
  //   },
  //   {
  //     to: "/file-based-sources",
  //     label: "File-based Data Sources",
  //     component: lazy(() => import("pages/module4/course/FileBased")),
  //     subLinks: [
  //       { id: "file-formats", label: "CSV, JSON, XML Files" },
  //       { id: "pandas-io", label: "Reading and Writing with Pandas" },
  //       { id: "large-files", label: "Handling Large Files" },
  //     ],
  //   },
  //   {
  //     to: "/databases",
  //     label: "Database Systems",
  //     component: lazy(() => import("pages/module4/course/Databases")),
  //     subLinks: [
  //       { id: "relational", label: "Relational Databases (SQL)" },
  //       { id: "nosql", label: "NoSQL Databases" },
  //     ],
  //   },
  //   {
  //     to: "/apis-web-scraping",
  //     label: "APIs and Web Scraping",
  //     component: lazy(() => import("pages/module4/course/APIsWebScraping")),
  //     subLinks: [
  //       { id: "restful-apis", label: "RESTful APIs" },
  //       { id: "web-scraping", label: "Web Scraping Basics" },
  //     ],
  //   },
  //   {
  //     to: "/real-time-streams",
  //     label: "Real-time Data Streams",
  //     component: lazy(() => import("pages/module4/course/RealTimeStreams")),
  //     subLinks: [
  //       { id: "intro-streaming", label: "Introduction to Streaming Data" },
  //       { id: "kafka", label: "Working with Apache Kafka" },
  //       { id: "processing-streams", label: "Processing Streams with Python" },
  //     ],
  //   },
  //   {
  //     to: "/cloud-based-sources",
  //     label: "Cloud-based Data Sources",
  //     component: lazy(() => import("pages/module4/course/CloudBasedSources")),
  //     subLinks: [
  //       { id: "object-storage", label: "Object Storage Systems" },
  //       { id: "data-warehouses", label: "Cloud Data Warehouses" },
  //       { id: "python-sdks", label: "Accessing Cloud Data with Python SDKs" },
  //     ],
  //   },
  //   {
  //     to: "/data-manipulation",
  //     label: "Data Manipulation with Pandas",
  //     component: lazy(() => import("pages/module4/course/DataManipulation")),
  //     subLinks: [
  //       { id: "dataframe-series", label: "DataFrame and Series Objects" },
  //       {
  //         id: "loading-cleaning",
  //         label: "Data Loading, Cleaning, and Preprocessing",
  //       },
  //       { id: "merging-joining", label: "Merging, Joining, and Concatenating" },
  //       { id: "grouping-aggregation", label: "Grouping and Aggregation" },
  //     ],
  //   },
  //   {
  //     to: "/data-quality",
  //     label: "Data Quality and Validation",
  //     component: lazy(() => import("pages/module4/course/DataQuality")),
  //     subLinks: [
  //       { id: "assessing-quality", label: "Assessing Data Quality" },
  //       { id: "profiling", label: "Data Profiling Techniques" },
  //       { id: "validation-checks", label: "Implementing Validation Checks" },
  //     ],
  //   },
  //   {
  //     to: "/CaseStudy4",
  //     label: "CaseStudy",
  //     component: lazy(() => import("pages/module4/course/CaseStudy")),
  //   },
  // ];

  const location = useLocation();
  const module = 4;
  return (
    <ModuleFrame
      module={4}
      isCourse={true}
      title="Module 4: Data Collection"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about the various aspects of data
              collection, including different data sources, techniques for
              retrieving and handling data, and important considerations for
              data quality and manipulation.
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

export default CourseDataCollection;
