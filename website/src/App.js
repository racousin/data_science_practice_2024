import React, { useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate, Link, useLocation } from "react-router-dom";
import { Navbar, Nav, Container } from "react-bootstrap";
import ReactGA from "react-ga4";

import Home from "./pages/Home";
import RepositoriesList from "pages/RepositoriesList";
import StudentsList from "./pages/StudentsList";
import Student from "./pages/Student";
import Teaching from "./pages/Teaching";

import PrerequistAandMethodologie from "pages/module0/PrerequistAandMethodologie";
import CourseGit from "pages/module1/CourseGit";
import ExerciseGit from "pages/module1/ExerciseGit";
import CoursePython from "pages/module2/CoursePython";
import ExercisePython from "pages/module2/ExercisePython";
import CourseDataScienceLandscape from "pages/module3/CourseDataScienceLandscape";
import ExerciseMLPipelineAndExploratoryDataAnalysis from "pages/module3/ExerciseDataScienceLandscape";
import CourseDataCollection from "pages/module4/CourseDataCollection";
import ExerciseDataCollection from "pages/module4/ExerciseDataCollection";
import CourseDataCleaningAndPreparation from "pages/module5/CourseDataCleaningAndPreparation";
import ExerciseDataCleaningAndPreparation from "pages/module5/ExerciseDataCleaningAndPreparation";
import CourseTabularModels from "pages/module6/CourseTabularModels";
import ExerciseTabularModels from "pages/module6/ExerciseTabularModels";
import CourseDeepLearningFundamentals from "pages/module7/CourseDeepLearningFundamentals";
import ExerciseDeepLearningFundamentals from "pages/module7/ExerciseDeepLearningFundamentals";
import CourseImageProcessing from "pages/module8/CourseImageProcessing";
import ExerciseImageProcessing from "pages/module8/ExerciseImageProcessing";
import CourseTextProcessing from "pages/module10/CourseTextProcessing";
import ExerciseTextProcessing from "pages/module10/ExerciseTextProcessing";
import CourseGenerativeModels from "pages/module11/CourseGenerativeModels";
import ExerciseGenerativeModels from "pages/module11/ExerciseGenerativeModels";
import CourseRecommendationSystems from "pages/module12/CourseRecommendationSystems";
import ExerciseRecommendationSystems from "pages/module12/ExerciseRecommendationSystems";
import CourseReinforcementLearning from "pages/module13/CourseReinforcementLearning";
import ExerciseReinforcementLearning from "pages/module13/ExerciseReinforcementLearning";
import CourseDocker from "pages/module14/CourseDocker";
import ExerciseDocker from "pages/module14/ExerciseDocker";
import CourseCloudIntegration from "pages/module15/CourseCloudIntegration";
import ExerciseCloudIntegration from "pages/module15/ExerciseCloudIntegration";

import PageToScrap from "pages/module4/course/PageToScrap";
import PageToScrapExercise from "pages/module4/course/PageToScrapExercise";

import Resources from "pages/Resources";

import "App.css";

import SearchNavbar from "components/SearchNavbar";
import SearchResultsPage from "pages/SearchResultsPage";

ReactGA.initialize("G-0VW1PJ0H17"); 

const RobotsTxt = () => {
  // Sitemap: https://example.com/sitemap.xml TODO
  return (
    <pre>
      {`User-agent: *
Allow: /
Disallow: /private/
Disallow: /admin/

User-agent: Googlebot
Allow: /

Crawl-delay: 5


`}
    </pre>
  );
};

function App() {
  return (
    <BrowserRouter>
      <Navbar
        bg="dark"
        variant="dark"
        expand="lg"
        sticky="top"
        className="navbar"
      >
        <Container>
          <Navbar.Brand as={Link} to="/">
            Teaching Portal
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link as={Link} to="/">
                Home
              </Nav.Link>
              <Nav.Link as={Link} to="/teaching">
                Teaching
              </Nav.Link>

              <Nav.Link as={Link} to="/repositories">
                Sessions Results
              </Nav.Link>
            </Nav>
            <SearchNavbar />
          </Navbar.Collapse>
        </Container>
      </Navbar>
      <Container fluid className="mt-3">
      <GoogleAnalyticsRouteTracker />
        <Routes>
        
          <Route path="/" element={<Home />} />
          <Route path="search" element={<SearchResultsPage />} />
          <Route path="repositories" element={<RepositoriesList />} />
          <Route path="students/:repositoryId" element={<StudentsList />} />
          <Route
            path="student/:repositoryId/:studentId"
            element={<Student />}
          />
          <Route path="teaching" element={<Teaching />} />
          <Route path="resources" element={<Resources />} />
          <Route path="robots.txt" element={<RobotsTxt />} />
          <Route
            path="module0/course"
            element={<PrerequistAandMethodologie />}
          />
          <Route path="module1/course/*" element={<CourseGit />} />
          <Route path="module1/exercise/*" element={<ExerciseGit />} />
          <Route path="module2/course/*" element={<CoursePython />} />
          <Route path="module2/exercise/*" element={<ExercisePython />} />
          <Route
            path="module3/course/*"
            element={<CourseDataScienceLandscape />}
          />
          <Route
            path="module3/exercise/*"
            element={<ExerciseMLPipelineAndExploratoryDataAnalysis />}
          />
          <Route path="module4/course/*" element={<CourseDataCollection />} />
          <Route
            path="module4/exercise/*"
            element={<ExerciseDataCollection />}
          />
          <Route path="page-to-scrap" element={<PageToScrap />} />
          <Route
            path="page-to-scrap-exercise/*"
            element={<PageToScrapExercise />}
          />
          <Route
            path="module5/course/*"
            element={<CourseDataCleaningAndPreparation />}
          />
          <Route
            path="module5/exercise/*"
            element={<ExerciseDataCleaningAndPreparation />}
          />
          <Route path="module6/course/*" element={<CourseTabularModels />} />
          <Route
            path="module6/exercise/*"
            element={<ExerciseTabularModels />}
          />
          <Route
            path="module7/course/*"
            element={<CourseDeepLearningFundamentals />}
          />
          <Route
            path="module7/exercise/*"
            element={<ExerciseDeepLearningFundamentals />}
          />
          <Route path="module8/course/*" element={<CourseImageProcessing />} />
          <Route
            path="module8/exercise/*"
            element={<ExerciseImageProcessing />}
          />

          <Route path="module10/course/*" element={<CourseTextProcessing />} />
          <Route
            path="module10/exercise/*"
            element={<ExerciseTextProcessing />}
          />
          <Route
            path="module11/course/*"
            element={<CourseGenerativeModels />}
          />
          <Route
            path="module11/exercise/*"
            element={<ExerciseGenerativeModels />}
          />
          <Route
            path="module12/course/*"
            element={<CourseRecommendationSystems />}
          />
          <Route
            path="module12/exercise/*"
            element={<ExerciseRecommendationSystems />}
          />
          <Route
            path="module13/course/*"
            element={<CourseReinforcementLearning />}
          />
          <Route
            path="module13/exercise/*"
            element={<ExerciseReinforcementLearning />}
          />
          <Route path="module14/course/*" element={<CourseDocker />} />
          <Route path="module14/exercise/*" element={<ExerciseDocker />} />
          <Route
            path="module15/course/*"
            element={<CourseCloudIntegration />}
          />
          <Route
            path="module15/exercise/*"
            element={<CourseCloudIntegration />}
          />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Container>
    </BrowserRouter>
  );
}

function GoogleAnalyticsRouteTracker() {
  const location = useLocation();

  useEffect(() => {
    ReactGA.send({ hitType: "pageview", page: location.pathname });
  }, [location]);

  return null;
}

export default App;
