import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ImageEnhancementTechniques = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Image Enhancement Techniques</h1>
      <p>
        In this section, you will explore techniques to enhance image quality
        for better analysis.
      </p>
      <Row>
        <Col>
          <h2>Histogram Equalization</h2>
          <p>
            Histogram equalization is a technique used to enhance the contrast
            of an image.
          </p>
          <CodeBlock
            code={`# Example of histogram equalization using OpenCV
equalized_img = cv2.equalizeHist(gray_img)`}
          />
          <h2>Noise Reduction</h2>
          <p>
            Noise reduction techniques can be used to remove unwanted noise from
            an image.
          </p>
          <CodeBlock
            code={`# Example of Gaussian blur using OpenCV
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# Example of median filtering using OpenCV
filtered_img = cv2.medianBlur(img, 5)`}
          />
          <h2>Contrast Enhancement</h2>
          <p>
            Contrast enhancement techniques can be used to improve the
            visibility of details in an image.
          </p>
          <CodeBlock
            code={`# Example of contrast enhancement using OpenCV
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 0     # Brightness control (0-100)
enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ImageEnhancementTechniques;
