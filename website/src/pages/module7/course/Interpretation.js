import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Interpretation = () => {
  return (
    <Container fluid>
      <h2 id="saliency-maps">Saliency Maps and Activation Maximization</h2>

      <p>
        Saliency maps and activation maximization are techniques used to
        visualize what a neural network is focusing on when making predictions.
      </p>

      <h3>Saliency Maps</h3>
      <p>
        Saliency maps highlight the parts of an input image that are most
        important for the network's prediction.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

def compute_saliency_maps(X, y, model):
    model.eval()
    X.requires_grad_()
    
    scores = model(X)
    scores = scores.gather(1, y.view(-1, 1)).squeeze()
    scores.backward(torch.ones_like(scores))
    
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    return saliency

# Example usage
model = models.resnet18(pretrained=True)
img = torch.randn(1, 3, 224, 224, requires_grad=True)
target = torch.tensor([282])  # Example class index

saliency = compute_saliency_maps(img, target, model)

plt.imshow(saliency[0].detach().cpu().numpy(), cmap='hot')
plt.axis('off')
plt.show()
        `}
      />

      <h3>Activation Maximization</h3>
      <p>
        Activation maximization generates an input that maximally activates a
        specific neuron or feature map.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

def activation_maximization(model, layer_index, filter_index, num_iterations=30):
    model.eval()
    
    input_image = torch.randn(1, 3, 224, 224, requires_grad=True, device='cuda')
    
    for _ in range(num_iterations):
        model.zero_grad()
        
        x = input_image
        for index, layer in enumerate(model.features):
            x = layer(x)
            if index == layer_index:
                break
        
        loss = -torch.mean(x[0, filter_index])
        loss.backward()
        
        input_image.data = input_image.data - 1e-1 * input_image.grad.data
        input_image.data = torch.clamp(input_image.data, 0, 1)
        input_image.grad.data.zero_()
    
    return input_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

# Example usage
model = models.vgg16(pretrained=True).cuda()
img = activation_maximization(model, layer_index=10, filter_index=0)

plt.imshow(img)
plt.axis('off')
plt.show()
        `}
      />

      <h2 id="layer-visualization">Layer Visualization</h2>

      <p>
        Layer visualization helps us understand what different layers in a
        neural network are learning.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

def visualize_layer(model, layer_num, num_filters=16):
    layer = model.features[layer_num]
    if isinstance(layer, nn.Conv2d):
        weights = layer.weight.data.cpu()
        fig, axs = plt.subplots(num_filters//4, 4, figsize=(12, 12))
        for i in range(num_filters):
            ax = axs[i//4, i%4]
            ax.imshow(weights[i, 0], cmap='gray')
            ax.axis('off')
        plt.show()

# Example usage
model = models.vgg16(pretrained=True)
visualize_layer(model, layer_num=0)  # Visualize first convolutional layer
        `}
      />

      <h2 id="tsne">t-SNE for High-Dimensional Data Visualization</h2>

      <p>
        t-SNE (t-Distributed Stochastic Neighbor Embedding) is a technique for
        dimensionality reduction that is particularly well suited for the
        visualization of high-dimensional datasets.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_features(model, dataloader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model.features(inputs)
            features.append(outputs.view(outputs.size(0), -1))
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)

def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=0)
    clustered = tsne.fit_transform(features)
    plt.figure(figsize=(10, 10))
    plt.scatter(clustered[:, 0], clustered[:, 1], c=labels)
    plt.colorbar()
    plt.show()

# Example usage
model = models.vgg16(pretrained=True)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

features, labels = get_features(model, dataloader)
visualize_tsne(features.cpu().numpy(), labels.cpu().numpy())
        `}
      />
    </Container>
  );
};

export default Interpretation;
