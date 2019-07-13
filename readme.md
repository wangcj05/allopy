Allopy
======

Allopy is a package used for optimizing portfolios. For documentation on usage and module API, check out the [documentation site](https://allopy.readthedocs.io).

Installation
------------

```bash
# pip
pip install allopy

# conda
conda install -c danielbok allopy
``` 

Simple Usage
------------

The listing below shows the sample usage with the `BaseOptimizer` class.

```python
from allopy.optimize import BaseOptimizer
from scipy.stats import multivariate_normal as mvn

assets_mean = [0.12, 0.04]  # asset mean returns vector
assets_std = [
    [0.04, 0.002], 
    [0.002, 0.0014]
]  # asset covariance matrix

# hypothetical returns series
returns = mvn.rvs(mean=assets_mean, cov=assets_std, size=500, random_state=88)

def objective(w):
    return (returns @ w).mean()


def constraint(w):
    # we need to convert the constraint to standard form. So c(w) - K <= 0
    return (returns @ w).std() - 0.1


prob = BaseOptimizer(2)  # initialize the optimizer with 2 asset classes

# set the objective function
prob.set_max_objective(objective) 

# set the inequality constraint function
prob.add_inequality_constraint(constraint)

# set lower and upper bounds to 0 and 1 for all free variables (weights)
prob.set_bounds(0, 1)

# set equality matrix constraint, Ax = b. Weights sum to 1
prob.add_equality_matrix_constraint([[1, 1]], [1])

sol = prob.optimize()
print('Solution: ', sol)
```

To find out more on how to use the other enchanced optimizers, checkout the [documentation](https://allopy.readthedocs.io).
