import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Regularization = () => {
  return (
    <Container fluid>
      <h2 id="l1-l2">L1 and L2 Regularization</h2>

      <p>
        Regularization techniques help prevent overfitting by adding a penalty
        term to the loss function, discouraging complex models.
      </p>

      <h3>L1 Regularization (Lasso)</h3>
      <p>
        L1 regularization adds the absolute value of the weights to the loss
        function:
      </p>
      <BlockMath math="L_{regularized} = L + \lambda \sum_{i} |w_i|" />
      <p>
        L1 regularization tends to produce sparse models by pushing some weights
        to exactly zero.
      </p>

      <h3>L2 Regularization (Ridge)</h3>
      <p>
        L2 regularization adds the squared value of the weights to the loss
        function:
      </p>
      <BlockMath math="L_{regularized} = L + \lambda \sum_{i} w_i^2" />
      <p>
        L2 regularization tends to distribute weight values more evenly,
        preventing any single feature from having a very large impact.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch.nn as nn

# L1 regularization
l1_lambda = 0.01
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = criterion(output, target) + l1_lambda * l1_norm

# L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        `}
      />

      <h2 id="dropout">Dropout</h2>

      <p>
        Dropout is a regularization technique where randomly selected neurons
        are ignored during training. This helps prevent complex co-adaptations
        on training data.
      </p>

      <p>
        During inference, all neurons are used, but their outputs are scaled by
        the dropout rate to maintain the expected output magnitude.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        `}
      />

      <h2 id="batch-normalization">Batch Normalization</h2>

      <p>
        Batch Normalization normalizes the inputs to each layer for each
        mini-batch. This helps to address the internal covariate shift problem
        and generally allows for higher learning rates and faster convergence.
      </p>

      <BlockMath math="\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x^{(k)}] + \epsilon}}" />

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
        `}
      />

      <h2 id="early-stopping">Early Stopping</h2>

      <p>
        Early stopping is a form of regularization used to prevent overfitting
        when training a learner with an iterative method, such as gradient
        descent. It stops training when the model performance on a validation
        dataset starts to degrade.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# Usage in training loop
early_stopping = EarlyStopping(patience=10, verbose=True)
for epoch in range(num_epochs):
    # ... training code ...
    val_loss = validate(model, val_loader)
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break
        `}
      />

      <p>
        This implementation of early stopping keeps track of the best model
        based on validation loss and stops training if the validation loss
        doesn't improve for a specified number of epochs.
      </p>
    </Container>
  );
};

export default Regularization;
