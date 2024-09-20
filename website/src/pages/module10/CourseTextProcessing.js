import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseTextProcessing = () => {
  const courseLinks = [
    // {
    //   to: "/introduction",
    //   label: "Introduction to Natural Language Processing",
    //   component: lazy(() => import("pages/module9/course/Introduction")),
    //   subLinks: [
    //     { id: "what-is-nlp", label: "What is NLP?" },
    //     { id: "applications", label: "Applications of NLP" },
    //     { id: "challenges", label: "Challenges in NLP" },
    //   ],
    // },
    // {
    //   to: "/preprocessing",
    //   label: "Text Preprocessing",
    //   component: lazy(() => import("pages/module9/course/Preprocessing")),
    //   subLinks: [
    //     { id: "tokenization", label: "Tokenization" },
    //     { id: "lowercasing", label: "Lowercasing and Normalization" },
    //     { id: "stemming", label: "Stemming and Lemmatization" },
    //     { id: "stop-words", label: "Stop Words Removal" },
    //     { id: "regex", label: "Regular Expressions in NLP" },
    //   ],
    // },
    // {
    //   to: "/representation",
    //   label: "Text Representation",
    //   component: lazy(() => import("pages/module9/course/Representation")),
    //   subLinks: [
    //     { id: "bow", label: "Bag of Words (BoW)" },
    //     {
    //       id: "tfidf",
    //       label: "Term Frequency-Inverse Document Frequency (TF-IDF)",
    //     },
    //     { id: "word-embeddings", label: "Word Embeddings (Word2Vec, GloVe)" },
    //     { id: "subword-embeddings", label: "Subword Embeddings (FastText)" },
    //   ],
    // },
    // {
    //   to: "/rnn",
    //   label: "Recurrent Neural Networks for NLP",
    //   component: lazy(() => import("pages/module9/course/RNN")),
    //   subLinks: [
    //     { id: "basic-rnn", label: "Basic RNN Architecture" },
    //     { id: "lstm", label: "Long Short-Term Memory (LSTM)" },
    //     { id: "gru", label: "Gated Recurrent Unit (GRU)" },
    //     { id: "bidirectional", label: "Bidirectional RNNs" },
    //   ],
    // },
    // {
    //   to: "/seq2seq",
    //   label: "Sequence-to-Sequence Models",
    //   component: lazy(() => import("pages/module9/course/Seq2Seq")),
    //   subLinks: [
    //     { id: "encoder-decoder", label: "Encoder-Decoder Architecture" },
    //     { id: "attention", label: "Attention Mechanism" },
    //     {
    //       id: "applications",
    //       label: "Applications (Machine Translation, Text Summarization)",
    //     },
    //   ],
    // },
    // {
    //   to: "/transformer",
    //   label: "Transformer Architecture",
    //   component: lazy(() => import("pages/module9/course/Transformer")),
    //   subLinks: [
    //     {
    //       id: "self-attention",
    //       label: "Self-Attention and Multi-Head Attention",
    //     },
    //     { id: "positional-encoding", label: "Positional Encoding" },
    //     { id: "encoder-decoder", label: "Transformer Encoder and Decoder" },
    //     { id: "bert", label: "BERT and its Variants" },
    //   ],
    // },
    // {
    //   to: "/transfer-learning",
    //   label: "Transfer Learning in NLP",
    //   component: lazy(() => import("pages/module9/course/TransferLearning")),
    //   subLinks: [
    //     { id: "fine-tuning", label: "Fine-tuning Pre-trained Models" },
    //     { id: "bert-tasks", label: "BERT for Various NLP Tasks" },
    //     {
    //       id: "other-models",
    //       label: "Other Transfer Learning Models (GPT, XLNet, RoBERTa)",
    //     },
    //   ],
    // },
    // {
    //   to: "/classification",
    //   label: "Text Classification",
    //   component: lazy(() => import("pages/module9/course/Classification")),
    //   subLinks: [
    //     { id: "sentiment-analysis", label: "Sentiment Analysis" },
    //     { id: "topic-classification", label: "Topic Classification" },
    //     { id: "spam-detection", label: "Spam Detection" },
    //   ],
    // },
    // {
    //   to: "/CaseStudy9",
    //   label: "CaseStudy",
    //   component: lazy(() => import("pages/module9/course/CaseStudy")),
    // },
    // {
    //   to: "/ner",
    //   label: "Named Entity Recognition (NER)",
    //   component: lazy(() => import("pages/module9/course/NER")),
    //   subLinks: [
    //     { id: "iob-tagging", label: "IOB Tagging" },
    //     { id: "crf", label: "CRF for NER" },
    //     { id: "bert-ner", label: "BERT for NER" },
    //   ],
    // },
    // {
    //   to: "/machine-translation",
    //   label: "Machine Translation",
    //   component: lazy(() => import("pages/module9/course/MachineTranslation")),
    //   subLinks: [
    //     { id: "statistical-mt", label: "Statistical Machine Translation" },
    //     { id: "neural-mt", label: "Neural Machine Translation" },
    //     { id: "bleu", label: "Evaluation Metrics (BLEU Score)" },
    //   ],
    // },
    // {
    //   to: "/text-generation",
    //   label: "Text Generation",
    //   component: lazy(() => import("pages/module9/course/TextGeneration")),
    //   subLinks: [
    //     { id: "language-modeling", label: "Language Modeling" },
    //     { id: "conditional-generation", label: "Conditional Text Generation" },
    //     { id: "beam-search", label: "Beam Search and Sampling Strategies" },
    //   ],
    // },
    // {
    //   to: "/qa-systems",
    //   label: "Question Answering Systems",
    //   component: lazy(() => import("pages/module9/course/QASystems")),
    //   subLinks: [
    //     { id: "types", label: "Types of QA Systems" },
    //     { id: "bert-qa", label: "BERT for QA" },
    //     { id: "evaluation", label: "Evaluating QA Systems" },
    //   ],
    // },
    // {
    //   to: "/summarization",
    //   label: "Text Summarization",
    //   component: lazy(() => import("pages/module9/course/Summarization")),
    //   subLinks: [
    //     { id: "extractive", label: "Extractive Summarization" },
    //     { id: "abstractive", label: "Abstractive Summarization" },
    //     { id: "rouge", label: "Evaluation Metrics (ROUGE Score)" },
    //   ],
    // },
  ];

  const location = useLocation();
  const module = 10;
  return (
    <ModuleFrame
      module={9}
      isCourse={true}
      title="Module 10: Text Processing"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              This module covers advanced topics in Natural Language Processing
              (NLP) using Deep Learning techniques.
            </p>
          </Row>
          <Row>
            <Col>
              <p>Last Updated: {"2024-09-20"}</p>
            </Col>
          </Row>
        </>
      )}
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseTextProcessing;
