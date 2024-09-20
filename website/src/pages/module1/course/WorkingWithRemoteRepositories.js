import React from "react";
import { Container, Row, Col, Nav, Tab, Image } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const WorkingWithRemoteRepositories = () => {
  return (
    <Container fluid>
      <Row>
        <Col>
          {/* Introduction to Remote Repositories */}
          <Row>
            <Col>
              <h3 id="why-remote-repositories">
                Why Remote Repositories are Useful
              </h3>
              <p>
                Remote repositories on platforms like GitHub allow developers to
                store versions of their projects online, facilitating
                collaboration, backup, and public sharing. They enable teams to
                work together on projects from different locations, track
                changes, merge contributions, and maintain a history of all
                modifications.
              </p>
            </Col>
          </Row>
          <Row className="justify-content-center">
            <Col xs={12} md={10} lg={8}>
              <div className="text-center">
                <Image
                  src="/assets/module1/Git_Remote_Workflow.png"
                  alt="Git_Remote_Workflow"
                  fluid
                />
                <p>Git_Remote_Workflow</p>
              </div>
            </Col>
          </Row>
          {/* Creating a Repository on GitHub */}
          <Row>
            <Col>
              <h3 id="creating-repository-github">
                Creating a Repository on GitHub
              </h3>
              <p>
                Creating a repository on GitHub is the first step toward remote
                project management. Here's how to set one up:
              </p>
              <ol>
                <li>Log in to your GitHub account.</li>
                <li>Navigate to the Repositories tab and click 'New'.</li>
                <li>
                  Enter a name for your repository and select the visibility
                  (public or private).
                </li>
                <li>
                  Optionally, initialize the repository with a README,
                  .gitignore, or license.
                </li>
                <li>Click 'Create repository'.</li>
              </ol>
            </Col>
          </Row>

          {/* Adding and Cloning Remote Repositories */}
          <Row>
            <Col>
              <h2>Connecting to Remote Repositories</h2>
              <ol>
                <li>Navigate to the Repository of interest.</li>
                <li>Click &#60;&#62; Code ▼.</li>
                <li>Select SSH.</li>
                <li>Copy the &#60;remote_repository_url&#62;</li>
              </ol>
              <div className="mytab">
                <Tab.Container defaultActiveKey="clone">
                  <Nav variant="pills" className="mb-3">
                    <Nav.Item>
                      <Nav.Link eventKey="clone" className="mx-1">
                        Clone Remote Repository
                      </Nav.Link>
                    </Nav.Item>
                    <Nav.Item>
                      <Nav.Link eventKey="add" className="mx-1">
                        Add to Existing Local Repository
                      </Nav.Link>
                    </Nav.Item>
                  </Nav>
                  <Tab.Content>
                    <Tab.Pane eventKey="add">
                      <h4>Add a Remote Repository</h4>
                      <p>
                        Link your existing local repository to a remote
                        repository on GitHub using the following command:
                      </p>
                      <CodeBlock
                        code={`git remote add origin <remote_repository_url>`}
                      />
                      <p>
                        This command sets the specified URL as 'origin', which
                        is the conventional name used by Git to reference the
                        primary remote.
                      </p>
                    </Tab.Pane>
                    <Tab.Pane eventKey="clone">
                      <h4>Clone a Remote Repository</h4>
                      <p>
                        To work on an existing project, clone the remote
                        repository with this command:
                      </p>
                      <CodeBlock code={`git clone <remote_repository_url>`} />
                      <p>
                        Cloning creates a local copy of the repository,
                        including all historical commits and branches.
                      </p>
                    </Tab.Pane>
                  </Tab.Content>
                </Tab.Container>
              </div>
            </Col>
          </Row>
          <Row>
            <Col>
              <h3 id="view-remote-repositories">View Remote Repositories</h3>
              <p>
                The <code>git remote -v</code> command is used to view all the
                remote repositories that your current repository knows about.
                Here's what the command does:
              </p>
              <ul>
                <li>
                  <strong>List Remotes:</strong> This command lists all remote
                  handles (short names) for the remote URLs configured.
                </li>
                <li>
                  <strong>Show URLs:</strong> Alongside each remote handle, the
                  command also shows the associated URLs, which can be either
                  fetch (data fetch from) or push (data send to) URLs.
                </li>
              </ul>
              <CodeBlock code="git remote -v" />
              <CodeBlock
                code={`$ git remote -v
origin	git@github.com:racousin/data_science_practice.git (fetch)
origin	git@github.com:racousin/data_science_practice.git (push)
`}
                language=""
              />
              <p>
                This output is particularly useful for verifying which remotes
                are set up for fetch and push operations, ensuring that you have
                the correct access paths for collaboration and deployment.
              </p>
            </Col>
          </Row>
          <Row>
            <Col>
              <h3 id="fetch-changes">Fetch Changes from a Remote Repository</h3>
              <p>
                The <code>git fetch origin</code> command is used to fetch
                branches, tags, and other data from a remote repository
                identified by 'origin'. This command prepares your local
                repository for a merge by updating your remote tracking branches
                with the latest changes from the remote without altering your
                current working directory.
              </p>
              <ul>
                <li>
                  <strong>Download Data:</strong> This command downloads
                  commits, files, and refs from a remote repository into your
                  local repo's working directory.
                </li>
                <li>
                  <strong>Update Tracking Branches:</strong>{" "}
                  <code>git fetch</code> updates your tracking branches under{" "}
                  <code>refs/remotes/origin/</code>, which represent the state
                  of your branches at the remote repository.
                </li>
                <li>
                  <strong>No Workspace Changes:</strong> Fetching does not
                  change your own local branches and does not modify your
                  current workspace. It fetches the data but leaves your current
                  branch unchanged, ensuring that your local development is not
                  automatically disrupted by remote changes.
                </li>
              </ul>
              <CodeBlock code="git fetch origin" />
              <p>
                After fetching, you may want to integrate these updates into
                your local branch, which involves an additional step:
              </p>
            </Col>
          </Row>
          <Row>
            <Col>
              <h3 id="merge-fetched-changes">Merge Fetched Changes</h3>
              <p>
                To merge the fetched changes into your current branch, you use
                the <code>git merge</code> command. This would typically involve
                merging the fetched branch (like <code>origin/main</code>) into
                your current branch:
              </p>
              <CodeBlock code="git merge origin/main" />
              <p>
                This command will merge the latest changes from the remote
                'main' branch into your current branch, allowing you to
                synchronize your local development with the latest updates from
                the remote repository.
              </p>
              <p>
                It's important to note that if there are any conflicts between
                the new changes and your local changes, you'll need to resolve
                them manually before completing the merge.
              </p>
            </Col>
          </Row>
          <Row>
            <Col>
              <h3 id="pull-changes">Pull Changes from a Remote Repository</h3>
              <p>
                The <code>git pull origin main</code> command combines two
                distinct operations performed by Git:
              </p>
              <ul>
                <li>
                  <strong>Fetch:</strong> First, <code>git pull</code> executes{" "}
                  <code>git fetch</code> which downloads content from the
                  specified remote repository under the branch named 'main'.
                  This step involves retrieving all the new commits, branches,
                  and files from the remote repository.
                </li>
                <li>
                  <strong>Merge:</strong> After fetching, Git automatically
                  attempts to merge the new commits into your current local
                  branch. If you are on the 'main' branch locally, it will merge
                  the changes from 'origin/main' into your local 'main' branch.
                </li>
              </ul>
              <CodeBlock code="git pull origin main" />
              <p>
                This command is crucial for keeping your local development
                environment updated with changes made by other collaborators in
                the repository. It ensures that you are working on the latest
                version of the project, reducing conflicts and inconsistencies.
              </p>
            </Col>
          </Row>
          <Row className="justify-content-center">
            <Col xs={12} md={10} lg={8}>
              <div className="text-center">
                <Image
                  src="/assets/module1/Git_Fetch_Merge_Pull.png"
                  alt="Git_Fetch_Merge_Pull"
                  fluid
                />
                <p>Git_Fetch_Merge_Pull</p>
              </div>
            </Col>
          </Row>
          <Row>
            <Col>
              <h3 id="push-changes">Push Changes to a Remote Repository</h3>
              <p>
                The <code>git push origin main</code> command sends the commits
                made on your local branch 'main' to the remote repository named
                'origin'. Here's how it works:
              </p>
              <ul>
                <li>
                  <strong>Upload Local Commits:</strong> This command pushes all
                  the commits from your local 'main' branch to the remote 'main'
                  branch managed by the 'origin' repository.
                </li>
                <li>
                  <strong>Update Remote Branch:</strong> If successful, the
                  remote 'main' branch will now reflect all the changes you've
                  made locally. This is essential for sharing your work with
                  other collaborators.
                </li>
                <li>
                  <strong>Permissions and Conflicts:</strong> You must have
                  write access to the remote repository, and your local branch
                  should be up-to-date with the remote changes to avoid
                  conflicts during the push.
                </li>
              </ul>
              <CodeBlock code="git push origin main" />
              <p>
                Using this command effectively allows teams to maintain a
                synchronized and collaborative development process, ensuring
                that all contributions are integrated and tracked in the remote
                repository.
              </p>
            </Col>
          </Row>

          <Row>
            <Col>
              <h3 id="example-case">Example case with Remote Repositories</h3>
              <p>
                Here's a practical example of working with a remote repository:
              </p>
              <ol>
                <li>
                  <strong>Create and Clone Repository:</strong> Follow the steps
                  above to create a repository on GitHub and clone it.
                </li>
                <li>
                  <strong>Add a File:</strong> Create a new file 'example.txt',
                  add some content to it, and save it in your project directory.
                </li>
                <li>
                  <strong>Stage the File:</strong> Run{" "}
                  <code>git add example.txt</code> to stage the file.
                </li>
                <li>
                  <strong>Commit the Change:</strong> Commit the staged file
                  with <code>git commit -m "Add example.txt"</code>.
                </li>
                <li>
                  <strong>Push the Commit:</strong> Push your commit to GitHub
                  with <code>git push origin main</code>.
                </li>
                <li>
                  <strong>Verify on GitHub:</strong> Check your GitHub
                  repository online to see the 'example.txt' file.
                </li>
                <li>
                  <strong>Make Changes on GitHub:</strong> Edit 'example.txt' on
                  GitHub and commit the changes online.
                </li>
                <li>
                  <strong>Pull Changes:</strong> Pull the changes back to your
                  local repository with <code>git pull origin main</code>.
                </li>
              </ol>
              <p>
                This workflow covers creating, modifying, and syncing changes
                between local and remote repositories, demonstrating the
                collaborative possibilities of Git and GitHub.
              </p>
            </Col>
          </Row>
        </Col>
      </Row>
    </Container>
  );
};

export default WorkingWithRemoteRepositories;
