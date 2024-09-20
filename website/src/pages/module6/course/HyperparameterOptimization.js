import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const HyperparameterOptimization = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Hyperparameter Optimization</h1>

      <section>
        <p>
          Hyperparameter optimization is the process of finding the best set of
          hyperparameters for a machine learning model. Hyperparameters are
          parameters that are not learned from the data but are set before the
          learning process begins. Optimizing these parameters can significantly
          improve model performance.
        </p>
      </section>

      <section>
        <h2 id="grid-search">Grid Search</h2>
        <p>
          Grid search is an exhaustive search through a manually specified
          subset of the hyperparameter space. It tries all possible combinations
          of the specified hyperparameter values.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Create a base model
svm = SVC(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
          `}
        />
      </section>

      <section>
        <h2 id="random-search">Random Search</h2>
        <p>
          Random search selects random combinations of hyperparameters. It's
          often more efficient than grid search, especially when only a few
          hyperparameters actually matter.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define the parameter distributions
param_dist = {
    'C': uniform(0.1, 100),
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': uniform(0.001, 1),
    'degree': randint(1, 6)
}

# Instantiate the random search model
random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_dist,
                                   n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Fit the random search to the data
random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: {:.2f}".format(random_search.best_score_))
print("Test set score: {:.2f}".format(random_search.score(X_test, y_test)))
          `}
        />
      </section>

      <section>
        <h2 id="bayesian-optimization">Bayesian Optimization</h2>
        <p>
          Bayesian optimization builds a probabilistic model of the function
          mapping from hyperparameter values to the objective evaluated on a
          validation set. It tries to balance exploration of unknown areas and
          exploitation of known good areas.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from skopt import BayesSearchCV

# Define the search space
search_spaces = {
    'C': (1e-6, 1e+6, 'log-uniform'),
    'gamma': (1e-6, 1e+1, 'log-uniform'),
    'degree': (1, 8),  # integer valued parameter
    'kernel': ['rbf', 'poly', 'sigmoid'],
}

# Instantiate the BayesSearchCV object
bayes_search = BayesSearchCV(
    estimator=svm,
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Fit the Bayesian optimization to the data
bayes_search.fit(X_train, y_train)

print("Best parameters found: ", bayes_search.best_params_)
print("Best cross-validation score: {:.2f}".format(bayes_search.best_score_))
print("Test set score: {:.2f}".format(bayes_search.score(X_test, y_test)))
          `}
        />
      </section>

      <section>
        <h2 id="genetic-algorithms">Genetic Algorithms</h2>
        <p>
          Genetic algorithms are inspired by the process of natural selection.
          They evolve a population of hyperparameter configurations over
          multiple generations to find the best performing set.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import cross_val_score
import numpy as np
from deap import base, creator, tools, algorithms

# This is a simplified example. In practice, you'd need to implement more complex genetic operations.

# Define the fitness function
def evaluate(individual):
    C, gamma = individual
    svm = SVC(C=C, gamma=gamma, random_state=42)
    return np.mean(cross_val_score(svm, X_train, y_train, cv=5)),

# Set up the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0.1, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the genetic algorithm
population = toolbox.population(n=50)
result, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

best_ind = tools.selBest(result, 1)[0]
print("Best individual is: %s\nwith fitness: %s" % (best_ind, best_ind.fitness))
          `}
        />
      </section>
    </Container>
  );
};

export default HyperparameterOptimization;
