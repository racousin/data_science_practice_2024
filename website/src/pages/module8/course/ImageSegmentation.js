import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ImageSegmentation = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Image Segmentation</h1>
      <p>
        In this section, you will learn about different image segmentation
        techniques.
      </p>
      <Row>
        <Col>
          <h2>Thresholding</h2>
          <p>
            Thresholding is a simple image segmentation technique that separates
            an image into two regions based on a threshold value.
          </p>
          <CodeBlock
            code={`# Example of thresholding using OpenCV
_, thresholded_img = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)`}
          />
          <h2>Region-Based Segmentation</h2>
          <p>
            Region-based segmentation techniques group pixels based on their
            similarity in color, texture, or other features.
          </p>
          <CodeBlock
            code={`# Example of region-based segmentation using OpenCV
kmeans = cv2.KMeans(n_clusters=3, randomState=0)
labels = kmeans.fit_predict(img.reshape(-1, 3))
segmented_img = labels.reshape(img.shape[:2])`}
          />
          <h2>Watershed Algorithm</h2>
          <p>
            The watershed algorithm is a popular image segmentation technique
            that segments an image based on the watershed theory.
          </p>
          <CodeBlock
            code={`# Example of watershed algorithm using OpenCV
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
markers = cv2.connectedComponents(sure_fg)[1]
markers = markers + 1
markers[unknown==255] = 0
markers = cv2.watershed(img, markers)
segmented_img = cv2.cvtColor(markers, cv2.COLOR_GRAY2BGR)`}
          />
          <h2>Semantic Segmentation using Deep Learning</h2>
          <p>
            Semantic segmentation using deep learning techniques can achieve
            more accurate and detailed segmentation results.
          </p>
          <CodeBlock
            code={`# Example of semantic segmentation using U-Net in TensorFlow
# Load the U-Net model
model = tf.keras.models.load_model('unet_model.h5')

# Preprocess the input image
input_img = preprocess_input(img)

# Perform semantic segmentation
output = model.predict(np.expand_dims(input_img, axis=0))
segmented_img = np.argmax(output[0], axis=-1)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ImageSegmentation;
