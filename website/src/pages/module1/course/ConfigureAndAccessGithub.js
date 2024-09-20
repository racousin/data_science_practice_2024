import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ConfigureAndAccessGithub = () => {
  const commands = {
    configUser: "git config --global user.name 'Your Username'",
    configEmail: "git config --global user.email 'your_email@example.com'",
    sshKeyGen: "ssh-keygen -t rsa -b 4096 -C 'your_email@example.com'",
    sshStart: "eval $(ssh-agent -s)",
    sshAdd: "ssh-add ~/.ssh/id_rsa",
    testConnection: "ssh -T git@github.com",
  };

  return (
    <Container fluid>
      <h3 id="create-github-account">Create a GitHub Account</h3>
      <p>
        GitHub is a platform for hosting and collaborating on software
        development projects using Git. Creating a GitHub account is the first
        step towards managing your projects online, contributing to other
        projects, and collaborating with other developers.
      </p>
      <Row>
        <Col md={12}>
          <ol>
            <li>
              Visit the GitHub homepage:{" "}
              <a
                href="https://www.github.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                www.github.com
              </a>
              .
            </li>
            <li>
              Click on the “Sign up” button in the upper-right corner of the
              homepage.
            </li>
            <li>Follow the steps!</li>
          </ol>
        </Col>
      </Row>
      <Row>
        <Col md={12}>
          <h3 id="configure-git">Configure Git with Your Information</h3>
          <p>
            Now that you have a GitHub account and Git installed, it’s time to
            configure your Git setup to securely connect and interact with
            GitHub.
          </p>
          <p>
            Start by setting your GitHub username and email address in Git,
            which will be used to identify the commits you make:
          </p>
          <CodeBlock code={commands.configUser} language="bash" />
          <CodeBlock code={commands.configEmail} language="bash" />
          <h3 id="connect-to-github-with-ssh">Connect to GitHub with SSH</h3>
          <h4>Generating a New SSH Key</h4>
          <p>
            Securely connect to GitHub with SSH by generating a new SSH key:
          </p>
          <CodeBlock code={commands.sshKeyGen} language="bash" />
          <CodeBlock
            showCopy={false}
            code={`$ ssh-keygen -t rsa -b 4096 -C 'username@email.com'
Generating public/private rsa key pair.
Enter file in which to save the key (/home/username/.ssh/id_rsa): 
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /home/username/.ssh/id_rsa
Your public key has been saved in /home/username/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:U508DNoGVSuUKX7KM2Y6+U8tBQujvvqulLsd7/ohS5Q username@email.com
The key's randomart image is:
+---[RSA 4096]----+
|        ..++.    |
|        .=o= o   |
|       .+.* B    |
|       o.=.+ .   |
|      E.So. .    |
|     +  B. o     |
|    o ==.oo .    |
|   . ++* o .     |
|    =*B*=..      |
+----[SHA256]-----+
`}
            language=""
          />
          <p>
            After generating the key, start the SSH agent in the background and
            add your new SSH key to it:
          </p>
          <CodeBlock code={commands.sshStart} language="bash" />
          <CodeBlock code={commands.sshAdd} language="bash" />
          <p>
            For detailed steps tailored to different operating systems, visit
            the{" "}
            <a
              href="https://docs.github.com/en/authentication/connecting-to-github-with-ssh"
              target="_blank"
              rel="noopener noreferrer"
            >
              official GitHub SSH guide
            </a>
            .
          </p>

          <h4>Add Your SSH Public Key to GitHub</h4>
          <p>
            To enable secure SSH access to your GitHub account, you need to add
            your SSH public key to your GitHub settings:
          </p>
          <ol>
            <li>Go to your GitHub account settings.</li>
            <li>Navigate to "SSH and GPG keys" under "Access."</li>
            <li>Click on "New SSH key" to add a new key.</li>
            <li>
              Paste your public key (<code>cat .ssh/id_rsa.pub</code>) into the
              field and save it.
            </li>
          </ol>
          <p>
            This step is essential for authenticating your future SSH sessions
            with GitHub.
          </p>

          <h4>Test Your SSH Connection</h4>
          <p>Verify that SSH is properly set up by connecting to GitHub:</p>
          <CodeBlock code={commands.testConnection} language="bash" />
          <CodeBlock
            code={`$ ssh -T git@github.com
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
`}
            language=""
            showCopy={false}
          />
          <p>
            If the connection is successful, you'll see a message confirming you
            are authenticated.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default ConfigureAndAccessGithub;
