import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const GenerativeAdversarialNetworks = () => {
  return (
    <Container fluid>
      <Row>
        <Col>
          <h1>Generative Adversarial Networks (GANs)</h1>
          <p>
            In this section, you will dive deep into GANs, one of the most
            popular types of generative models.
          </p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>The Architecture of GANs: Generator and Discriminator</h2>
          <p>
            GANs consist of two parts: a generator and a discriminator. The
            generator is responsible for generating new data instances that
            resemble the training data, while the discriminator is responsible
            for distinguishing between real and fake data. The two parts are
            trained together in a minimax game, where the generator tries to
            fool the discriminator and the discriminator tries to correctly
            classify the data.
          </p>
          <CodeBlock
            code={`
// GAN architecture

const generator = tf.sequential();
generator.add(tf.layers.dense({inputShape: [latentDim], units: 128, activation: 'relu'}));
generator.add(tf.layers.dense({units: 128, activation: 'relu'}));
generator.add(tf.layers.dense({units: 784, activation: 'tanh'}));

const discriminator = tf.sequential();
discriminator.add(tf.layers.dense({inputShape: [784], units: 128, activation: 'relu'}));
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
          <h2>Training Challenges and Strategies to Stabilize GAN Training</h2>
          <p>
            GANs are notoriously difficult to train, and there are several
            challenges that can arise during the training process. In this
            section, you will learn about some of these challenges and
            strategies to stabilize GAN training.
          </p>
          <CodeBlock
            code={`
// GAN training

const dReal = discriminator.predict(x);
const dFake = discriminator.predict(g);

const dLoss = dFake.sub(dReal).mean();

discriminator.trainOnBatch(x, [1]);
discriminator.trainOnBatch(g, [0]);

generator.trainOnBatch(z, dFake);
            `}
            language="js"
            showLineNumbers={true}
          />
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Variants of GANs: DCGAN, CGAN, and CycleGAN</h2>
          <p>
            There are several variants of GANs that have been developed to
            address specific challenges or to improve the performance of GANs.
            In this section, you will learn about some of these variants,
            including DCGAN, CGAN, and CycleGAN.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default GenerativeAdversarialNetworks;
