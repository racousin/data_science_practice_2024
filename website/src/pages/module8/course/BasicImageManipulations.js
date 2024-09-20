import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const BasicImageManipulations = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Basic Image Manipulations</h1>
      <p>
        In this section, you will learn how to perform basic image manipulations
        using Python.
      </p>
      <Row>
        <Col>
          <h2>Reading, Displaying, and Saving Images</h2>
          <p>
            Libraries like PIL (Python Imaging Library) and OpenCV can be used
            to read, display, and save images.
          </p>
          <CodeBlock
            code={`# Example of reading, displaying, and saving an image using PIL
from PIL import Image

# Read the image
img = Image.open('image.jpg')

# Display the image
img.show()

# Save the image
img.save('output.jpg')`}
          />
          <h2>Image Operations</h2>
          <p>
            Basic image operations include resizing, cropping, rotating, and
            flipping images.
          </p>
          <CodeBlock
            code={`# Example of resizing an image using PIL
img = img.resize((width, height))

# Example of cropping an image using OpenCV
cropped_img = img[y:y+h, x:x+w]

# Example of rotating an image using OpenCV
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Example of flipping an image using OpenCV
flipped_img = cv2.flip(img, 1)  # 1 for horizontal flip, 0 for vertical flip`}
          />
          <h2>Color Space Transformations</h2>
          <p>
            Color space transformations can be used to convert images between
            different color models.
          </p>
          <CodeBlock
            code={`# Example of converting an image from RGB to Grayscale using OpenCV
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Example of converting an image from BGR to RGB using OpenCV
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default BasicImageManipulations;
