import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const OtherGenerativeModels = () => {
  return (
    <Container fluid>
      <Row>
        <Col>
          <h1>Other Generative Models</h1>
          <p>
            In this section, you will study other types of generative models
            beyond VAEs and GANs.
          </p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>PixelRNN and PixelCNN for Generating Images Pixel by Pixel</h2>
          <p>
            PixelRNN and PixelCNN are types of generative models that can
            generate images pixel by pixel. They use a recurrent or
            convolutional architecture to model the joint probability
            distribution of the pixels in the image.
          </p>
          <CodeBlock
            code={`
// PixelRNN architecture

const pixelRNN = tf.sequential();
pixelRNN.add(tf.layers.lstm({inputShape: [numRows * numCols, numChannels], units: 128}));
pixelRNN.add(tf.layers.dense({units: numRows * numCols * numChannels, activation: 'softmax'}));
            `}
            language="js"
            showLineNumbers={true}
          />
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Auto-regressive Models like GPT for Text Generation</h2>
          <p>
            Auto-regressive models like GPT are types of generative models that
            can generate coherent and meaningful text. They use a transformer
            architecture to model the conditional probability distribution of
            the words in the text.
          </p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Energy-based Models for Unsupervised Learning</h2>
          <p>
            Energy-based models are types of generative models that can learn to
            model the underlying distribution of the data in an unsupervised
            manner. They use an energy function to model the probability density
            of the data.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default OtherGenerativeModels;
