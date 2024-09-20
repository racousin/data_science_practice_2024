import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseGit = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/module1/course/Introduction")),
    },
    {
      to: "/installing-git",
      label: "Installing Git",
      component: lazy(() => import("pages/module1/course/InstallingGit")),
      subLinks: [
        { id: "mac", label: "Mac" },
        { id: "windows", label: "Windows" },
        { id: "linux", label: "Linux" },
      ],
    },
    {
      to: "/first-steps-with-git",
      label: "First Steps with Git",
      component: lazy(() => import("pages/module1/course/FirstStepsWithGit")),
      subLinks: [
        { id: "creating-repository", label: "Creating a Git repository" },
        { id: "staging-changes", label: "Staging Changes" },
        { id: "committing-changes", label: "Committing Changes" },
        { id: "viewing-commit-history", label: "Viewing the Commit History" },
        { id: "undoing-changes", label: "Undoing Changes" },
      ],
    },
    {
      to: "/configure-and-access-github",
      label: "Configure And Access Github",
      component: lazy(() =>
        import("pages/module1/course/ConfigureAndAccessGithub")
      ),
      subLinks: [
        { id: "create-github-account", label: "Create GitHub Account" },
        {
          id: "configure-git",
          label: "Configure Git",
        },
        {
          id: "connect-to-github-with-ssh",
          label: "Connect to GitHub with SSH",
        },
      ],
    },
    {
      to: "/working-with-remote-repositories",
      label: "Working with Remote Repositories",
      component: lazy(() =>
        import("pages/module1/course/WorkingWithRemoteRepositories")
      ),
      subLinks: [
        {
          id: "why-remote-repositories",
          label: "Why Remote Repositories?",
        },
        {
          id: "creating-repository-github",
          label: "Creating a Repository on GitHub",
        },
        {
          id: "connecting-remote-repository",
          label: "Connecting to a Remote Repository",
        },
        {
          id: "view-remote-repositories",
          label: "View Remote Repositories",
        },
        {
          id: "fetch-changes",
          label: "Fetch Changes",
        },
        {
          id: "merge-fetched-changes",
          label: "Merge Fetched Changes",
        },
        {
          id: "pull-changes",
          label: "Pull Changes",
        },
        {
          id: "push-changes",
          label: "Push Changes",
        },
        {
          id: "example-case",
          label: "Example Case",
        },
      ],
    },

    {
      to: "/branching-and-merging",
      label: "Branching and Merging",
      component: lazy(() => import("pages/module1/course/BranchingAndMerging")),
      subLinks: [
        { id: "working-with-branches", label: "Working with Branches" },
        { id: "merging-branches", label: "Merging Branches" },
        { id: "resolving-merge-conflicts", label: "Resolving Merge Conflicts" },
        { id: "example-case", label: "Example: Adding a Feature" },
      ],
    },
    {
      to: "/collaborating",
      label: "Collaborating",
      component: lazy(() => import("pages/module1/course/Collaborating")),
      subLinks: [
        {
          id: "collaborating-with-others-using-github",
          label: "Using GitHub for Collaboration",
        },
        { id: "git-workflows", label: "Git Workflows" },
        { id: "peer-reviews", label: "Peer Reviews" },
      ],
    },
    {
      to: "/best-practices-and-resources",
      label: "Best Practices and Resources",
      component: lazy(() =>
        import("pages/module1/course/BestPracticesAndResources")
      ),
      subLinks: [
        {
          id: "good-practices",
          label: "Good Practices",
        },
        {
          id: "merge-vs-rebase",
          label: "Merge versus Rebase",
        },
        {
          id: "merge-vs-rebase",
          label: "Merge versus Rebase",
        },
        {
          id: "git-tagging",
          label: "Git tag",
        },
        {
          id: "using-git-stash",
          label: "Git stash",
        },
        {
          id: "ci-cd-github-actions",
          label: "CI/CD and GitHub Actions",
        },
        {
          id: "resources",
          label: "Resources",
        },
      ],
    },
  ];

  const location = useLocation();
  const module = 1;
  return (
    <ModuleFrame
      module={1}
      isCourse={true}
      title="Module 1: Git"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              Learn how to use Git for version control and GitHub for
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
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseGit;
