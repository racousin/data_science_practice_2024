import React, { useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Button } from "react-bootstrap";

const CodeBlock = ({ code, language = "bash", showCopy = true }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`position-relative ${!showCopy ? "terminal-result" : ""}`}>
      {showCopy && (
        <CopyToClipboard text={code} onCopy={handleCopy}>
          <Button
            variant="outline-secondary"
            size="sm"
            className="position-absolute top-0 end-0 m-2"
          >
            {copied ? "Copied!" : "Copy"}
          </Button>
        </CopyToClipboard>
      )}
      <SyntaxHighlighter language={language} style={a11yDark}>
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

export default CodeBlock;
