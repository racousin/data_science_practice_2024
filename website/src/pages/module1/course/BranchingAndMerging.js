import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const BranchingAndMerging = () => {
  return (
    <Container fluid>
      <h2>Branching and Merging</h2>
      <p>
        Branching and merging are vital features of Git that facilitate
        simultaneous and non-linear development among teams. Branching allows
        multiple developers to work on different features at the same time
        without interfering with each other, while merging brings those changes
        together into a single branch.
      </p>

      {/* Working with Branches */}
      <Row>
        <Col>
          <h3 id="working-with-branches">Working with Branches</h3>
          <p></p>
          <ol>
            <li>
              <strong>List all local Branches:</strong>
              <CodeBlock code={`git branch`} />
              <CodeBlock
                code={`$ git branch
  master
  my_branch1
* my_branch2
  my_branch3`}
                language=""
              />
              The <code>*</code> indicates the current branch.
            </li>

            <li>
              <strong>Create a Branch:</strong> Use{" "}
              <CodeBlock code={`git checkout -b newbranch`} />
              <CodeBlock
                code={`$ git checkout -b newbranch
Switched to a new branch 'newbranch'
`}
                language=""
              />
            </li>
          </ol>
        </Col>
      </Row>

      {/* Merging Branches */}
      <Row className="mt-4">
        <Col>
          <h3 id="merging-branches">Merging Branches</h3>
          <p>
            Once development on a branch is complete, the changes can be merged
            back into the main branch (e.g. 'main'). Here are different merge
            types:
          </p>
          <ul>
            <li>
              <strong>Fast-forward Merge:</strong>
              <p>
                Occurs when the target branch hasn't diverged from the source
                branch. Git simply moves the pointer forward.
              </p>
              <CodeBlock
                code={`git checkout main
git merge feature-branch`}
              />
              <CodeBlock
                code={`$ git checkout main
Switched to branch 'main'
$ git merge feature-branch
Updating 22a36a3..3951f63
Fast-forward
 example.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)`}
                language=""
              />
            </li>
            <li>
              <strong>Three-way Merge:</strong>
              <p>
                Occurs when the target branch has diverged from the source
                branch. Git creates a new commit to merge the histories.
              </p>
              <CodeBlock
                code={`git checkout main
git merge feature-branch`}
              />
              <CodeBlock
                code={`$ git checkout main
Switched to branch 'main'
$ git merge feature-branch
Merge made by the 'recursive' strategy.
 example.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)`}
                language=""
              />
            </li>
            <li>
              <strong>Squash Merge:</strong>
              <p>
                Combines all changes from the source branch into a single commit
                in the target branch.
              </p>
              <CodeBlock
                code={`git checkout main
git merge --squash feature-branch
git commit -m "Squashed feature-branch changes"`}
              />
            </li>
            <li>
              <strong>Rebase:</strong>
              <p>
                Moves the entire feature branch to begin on the tip of the main
                branch, effectively incorporating all new commits in main.
              </p>
              <CodeBlock
                code={`git checkout feature-branch
git rebase main
git checkout main
git merge feature-branch`}
              />
            </li>
            <li>
              <strong>No-fast-forward Merge:</strong>
              <p>
                Forces a new merge commit even when a fast-forward merge is
                possible. Useful for maintaining a record of merges.
              </p>
              <CodeBlock
                code={`git checkout main
git merge --no-ff feature-branch`}
              />
            </li>
          </ul>
        </Col>
      </Row>
      <Row className="justify-content-center">
        <Col xs={12} md={10} lg={8}>
          <div className="text-center">
            <Image
              src="/assets/module1/Git_Fast-forward_Merge.png"
              alt="Git_Fast-forward_Merge"
              fluid
            />
            <p>Git_Fast-forward_Merge</p>
          </div>
        </Col>
      </Row>
      <Row className="justify-content-center">
        <Col xs={12} md={10} lg={8}>
          <div className="text-center">
            <Image
              src="/assets/module1/Git_Rebase_Merge.png"
              alt="Git_Rebase_Merge"
              fluid
            />
            <p>Git_Rebase_Merge</p>
          </div>
        </Col>
      </Row>
      <Row className="justify-content-center">
        <Col xs={12} md={10} lg={8}>
          <div className="text-center">
            <Image
              src="/assets/module1/Git_Squash_Merge.png"
              alt="Git_Squash_Merge"
              fluid
            />
            <p>Git_Squash_Merge</p>
          </div>
        </Col>
      </Row>
      <Row className="justify-content-center">
        <Col xs={12} md={10} lg={8}>
          <div className="text-center">
            <Image
              src="/assets/module1/Git_Three-way_Merge.png"
              alt="Git_Three-way_Merge"
              fluid
            />
            <p>Git_Three-way_Merge</p>
          </div>
        </Col>
      </Row>
      {/* Merging Branches */}
      <Row className="mt-4">{/* ... (existing code) ... */}</Row>

      {/* Advantages of Merge Strategies */}
      <Row className="mt-4">
        <Col>
          <h4>Advantages of Different Merge Strategies</h4>
          <table className="table table-bordered">
            <thead>
              <tr>
                <th>Merge Type</th>
                <th>Advantages</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Fast-forward</td>
                <td>
                  - Simplest and cleanest history
                  <br />
                  - No additional merge commits
                  <br />- Preserves linear history
                </td>
              </tr>
              <tr>
                <td>Three-way</td>
                <td>
                  - Preserves complete history of both branches
                  <br />
                  - Clearly shows where branches diverged and merged
                  <br />- Useful for complex feature integrations
                </td>
              </tr>
              <tr>
                <td>Squash</td>
                <td>
                  - Simplifies feature history into a single commit
                  <br />
                  - Keeps main branch history clean and concise
                  <br />- Easier to revert entire features if needed
                </td>
              </tr>
              <tr>
                <td>Rebase</td>
                <td>
                  - Creates a linear, clean history
                  <br />
                  <br />- Avoids unnecessary merge commits
                </td>
              </tr>
              <tr>
                <td>No-fast-forward</td>
                <td>
                  - Always creates a merge commit
                  <br />
                  - Preserves branch structure and merge points
                  <br />- Useful for tracking when and where merges occurred
                </td>
              </tr>
            </tbody>
          </table>
        </Col>
      </Row>
      {/* Resolving Merge Conflicts */}
      <Row className="mt-4">
        <Col>
          <h3 id="resolving-merge-conflicts">Resolving Merge Conflicts</h3>
          <p>
            Conflicts occur when the same parts of the same file are changed in
            different branches:
          </p>
          <ol>
            <li>
              <strong>Identify Conflicts:</strong> During a merge, Git will tell
              you if there are conflicts that need manual resolution.
              <CodeBlock
                code={`$ git merge newbranch
Auto-merging example.txt
CONFLICT (content): Merge conflict in example.txt
Automatic merge failed; fix conflicts and then commit the result.
$ git status
On branch main
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)
	both modified:   example.txt
`}
                language=""
                showCopy={false}
              />
            </li>
            <li>
              <strong>Edit Files:</strong> Open the conflicted files and make
              the necessary changes to resolve conflicts.
            </li>
            <li>
              <strong>Mark as Resolved:</strong> Use <code>git add</code> on the
              resolved files to mark them as resolved.
              <CodeBlock code={`git add example.txt`} />
            </li>
            <li>
              <strong>Complete the Merge:</strong> Use <code>git commit</code>{" "}
              to complete the merge.
              <CodeBlock
                code={`git commit -m "Resolved merge conflict by including both suggestions."`}
              />
            </li>
          </ol>
        </Col>
      </Row>
      {/* Concrete Example of Branching and Merging */}
      <Row className="mt-4">
        <Col>
          <h3 id="example-case">Example: Adding a Feature via Branch</h3>
          <p>
            Imagine you are working on a project and need to add a new feature
            without disrupting the main development line. Here’s how you can
            handle it with Git branching and merging:
          </p>
          <ol>
            <li>
              <strong>Create a Feature Branch:</strong> Suppose you want to add
              a new login feature. You would start by creating a new branch
              dedicated to this feature.
              <CodeBlock code={`git branch login-feature`} />
            </li>
            <li>
              <strong>Switch to the Feature Branch:</strong> Move to the
              'login-feature' branch to work on this feature.
              <CodeBlock code={`git checkout login-feature`} />
            </li>
            <li>
              <strong>Develop the Feature:</strong> Make all necessary changes
              for the new feature. For example, create new files or modify
              existing ones, test the feature, etc.
              <CodeBlock
                code={`git add .\ngit commit -m "Add login feature"`}
              />
            </li>
            <li>
              <strong>Switch Back to Main Branch:</strong> Once the feature
              development is complete and tested, switch back to the main branch
              to prepare for merging.
              <CodeBlock code={`git checkout main`} />
            </li>
            <li>
              <strong>Merge the Feature Branch:</strong> Merge the changes from
              'login-feature' into 'main'. Assuming no conflicts, this merge
              will integrate the new feature into the main project.
              <CodeBlock code={`git merge login-feature`} />
            </li>
            <li>
              <strong>Delete the Feature Branch:</strong> After the feature has
              been successfully merged, you can delete the branch to keep the
              repository clean.
              <CodeBlock code={`git branch -d login-feature`} />
            </li>
          </ol>
          <p>
            This workflow keeps the main line stable while allowing development
            of new features in parallel. It also ensures that any ongoing work
            is not affected by the new changes until they are fully ready to be
            integrated.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default BranchingAndMerging;
