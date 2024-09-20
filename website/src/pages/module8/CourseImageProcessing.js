import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseImageProcessing = () => {
  const courseLinks = []
  // const courseLinks = [
  //   {
  //     to: "/introduction",
  //     label: "Introduction to Computer Vision",
  //     component: lazy(() => import("pages/module8/course/Introduction")),
  //     subLinks: [
  //       { id: "history", label: "Brief history of computer vision" },
  //       { id: "challenges", label: "Challenges in computer vision" },
  //       {
  //         id: "applications",
  //         label: "Applications of deep learning in computer vision",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/image-basics",
  //     label: "Image Basics and Preprocessing",
  //     component: lazy(() => import("pages/module8/course/ImageBasics")),
  //     subLinks: [
  //       { id: "representation", label: "Image representation in computers" },
  //       { id: "color-spaces", label: "Color spaces (RGB, HSV, etc.)" },
  //       {
  //         id: "normalization",
  //         label: "Image normalization and standardization",
  //       },
  //       { id: "augmentation", label: "Data augmentation techniques" },
  //     ],
  //   },
  //   {
  //     to: "/cnn",
  //     label: "Convolutional Neural Networks (CNNs)",
  //     component: lazy(() => import("pages/module7/course/CNN")),
  //     subLinks: [
  //       { id: "convolution", label: "Convolution operation and intuition" },
  //       { id: "pooling", label: "Pooling layers" },
  //     ],
  //   },
  //   {
  //     to: "/cnn_archi",
  //     label: "CNN Architecture",
  //     component: lazy(() => import("pages/module8/course/CNN_archi")),
  //     subLinks: [
  //       { id: "architecture", label: "Review of CNN architecture" },
  //       {
  //         id: "popular-models",
  //         label: "Popular CNN architectures (AlexNet, VGG, ResNet, Inception)",
  //       },
  //       { id: "transfer-learning", label: "Transfer learning and fine-tuning" },
  //     ],
  //   },
  //   {
  //     to: "/classification",
  //     label: "Image Classification",
  //     component: lazy(() => import("pages/module8/course/Classification")),
  //     subLinks: [
  //       {
  //         id: "multi-class",
  //         label: "Multi-class and multi-label classification",
  //       },
  //       {
  //         id: "implementation",
  //         label: "Implementing image classification with PyTorch",
  //       },
  //       {
  //         id: "evaluation",
  //         label: "Evaluation metrics for image classification",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/object-detection",
  //     label: "Object Detection",
  //     component: lazy(() => import("pages/module8/course/ObjectDetection")),
  //     subLinks: [
  //       {
  //         id: "rcnn",
  //         label: "Region-based CNNs (R-CNN, Fast R-CNN, Faster R-CNN)",
  //       },
  //       { id: "ssd-yolo", label: "Single Shot Detectors (SSD, YOLO)" },
  //       {
  //         id: "implementation",
  //         label: "Implementing object detection with PyTorch",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/semantic-segmentation",
  //     label: "Semantic Segmentation",
  //     component: lazy(() =>
  //       import("pages/module8/course/SemanticSegmentation")
  //     ),
  //     subLinks: [
  //       { id: "fcn", label: "Fully Convolutional Networks (FCN)" },
  //       { id: "unet", label: "U-Net architecture" },
  //       {
  //         id: "implementation",
  //         label: "Implementing semantic segmentation with PyTorch",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/instance-segmentation",
  //     label: "Instance Segmentation",
  //     component: lazy(() =>
  //       import("pages/module8/course/InstanceSegmentation")
  //     ),
  //     subLinks: [
  //       { id: "mask-rcnn", label: "Mask R-CNN" },
  //       {
  //         id: "implementation",
  //         label: "Implementing instance segmentation with PyTorch",
  //       },
  //     ],
  //   },
  //   {
  //     to: "/CaseStudy8",
  //     label: "CaseStudy",
  //     component: lazy(() => import("pages/module8/course/CaseStudy")),
  //   },
  // ];

  const location = useLocation();
  const module = 8;
  return (
    <ModuleFrame
      module={8}
      isCourse={true}
      title="Module 8: Image Processing with Deep Learning"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              This module covers advanced topics in image processing using deep
              learning techniques, from the basics of computer vision to complex
              tasks like instance segmentation.
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

export default CourseImageProcessing;
