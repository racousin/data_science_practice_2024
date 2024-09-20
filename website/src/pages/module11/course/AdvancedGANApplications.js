import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const AdvancedGANApplications = () => {
  return (
    <Container fluid>
      <Row>
        <Col>
          <h1>Advanced GAN Applications</h1>
          <p>
            In this section, you will explore advanced applications and
            improvements of GAN technology.
          </p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Conditional GANs for Controlled Generation</h2>
          <p>
            Conditional GANs (cGANs) are a type of GAN that can generate new
            data instances conditioned on some input information. This allows
            for more controlled and targeted generation of data.
          </p>
          <CodeBlock
            code={`
// cGAN architecture

const generator = tf.sequential();
generator.add(tf.layers.dense({inputShape: [latentDim + labelDim], units: 128, activation: 'relu'}));
generator.add(tf.layers.dense({units: 128, activation: 'relu'}));
generator.add(tf.layers.dense({units: 784, activation: 'tanh'}));

const discriminator = tf.sequential();
discriminator.add(tf.layers.dense({inputShape: [784 + labelDim], units: 128, activation: 'relu'}));
discriminator.add(tf.layers.dense({units: 128, activation: 'relu'}));
discriminator.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
            `}
            language="js"
            showLineNumbers={true}
          />
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>StyleGANs for High-Resolution Image Synthesis</h2>
          <p>
            StyleGANs are a type of GAN that can generate high-resolution and
            realistic images. They use a novel architecture that separates the
            style and content of the image, allowing for more fine-grained
            control over the generation process.
          </p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>
            GANs for Creating Synthetic Data in Healthcare and Other Fields
          </h2>
          <p>
            GANs can be used to create synthetic data that can be used in a
            variety of fields, including healthcare, finance, and marketing.
            This can be especially useful when the real data is scarce or
            sensitive.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default AdvancedGANApplications;
