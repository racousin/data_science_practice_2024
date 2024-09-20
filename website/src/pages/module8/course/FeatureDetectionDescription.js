import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const FeatureDetectionDescription = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Feature Detection and Description</h1>
      <p>
        In this section, you will understand methods to detect and describe
        important features in images.
      </p>
      <Row>
        <Col>
          <h2>Edge Detection</h2>
          <p>
            Edge detection is a technique used to identify the boundaries of
            objects in an image.
          </p>
          <h3>Sobel Edge Detection</h3>
          <CodeBlock
            code={`# Example of Sobel edge detection using OpenCV
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobel_img = cv2.sqrt(sobelx**2 + sobely**2)`}
          />
          <h3>Canny Edge Detection</h3>
          <CodeBlock
            code={`# Example of Canny edge detection using OpenCV
edges = cv2.Canny(img, threshold1=100, threshold2=200)`}
          />
          <h2>Corner Detection</h2>
          <p>
            Corner detection is a technique used to identify the corners or
            interest points in an image.
          </p>
          <h3>Harris Corner Detection</h3>
          <CodeBlock
            code={`# Example of Harris corner detection using OpenCV
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)`}
          />
          <h3>Shi-Tomasi Corner Detection</h3>
          <CodeBlock
            code={`# Example of Shi-Tomasi corner detection using OpenCV
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)`}
          />
          <h2>Feature Descriptors</h2>
          <p>
            Feature descriptors are used to describe the features detected in an
            image.
          </p>
          <h3>SIFT (Scale-Invariant Feature Transform)</h3>
          <CodeBlock
            code={`# Example of SIFT feature detection and description using OpenCV
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)`}
          />
          <h3>SURF (Speeded Up Robust Features)</h3>
          <CodeBlock
            code={`# Example of SURF feature detection and description using OpenCV
surf = cv2.SURF_create()
keypoints, descriptors = surf.detectAndCompute(img, None)`}
          />
          <h3>ORB (Oriented FAST and Rotated BRIEF)</h3>
          <CodeBlock
            code={`# Example of ORB feature detection and description using OpenCV
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default FeatureDetectionDescription;
