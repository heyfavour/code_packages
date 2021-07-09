from ray import tune

"""


def calc_param(step, alpha, beta):
    # if step<5:step = step*10
    return 0.1 + alpha  + beta * 0.1 +step
    # return (0.1 + alpha * step / 100) ** (-1) + beta * 0.1


def training_function(config):
    # Hyperparameters
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        intermediate_score = calc_param(step, alpha, beta)
        # Feed the score back back to Tune.
        tune.report(mean_loss=intermediate_score)#以最后汇报的数据为准


config = {
    "alpha": tune.grid_search([0.001, 0.01, 0.1]),
    # "beta": tune.choice([1, 2, 3, 4, 5]),#随机sample
    "beta": tune.grid_search([1]),#随机sample
}
analysis = tune.run(training_function, config=config)

print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
print("========================================================================")
df = analysis.results_df
print(df)
"""

"""
def objective(x, a, b):
    return a * (x ** 0.5) + b


def trainable(config):
    # config (dict): A dict of hyperparameters.
    for x in range(20):
        score = objective(x, config["a"], config["b"])
        tune.report(score=score)  # This sends the score to Tune.

tune.run(trainable)
tune.run(trainable, num_samples=10)

space = {"x": tune.uniform(0, 1)}
tune.run(trainable, config=space, num_samples=10)
"""
# ------------------------------------------------------------------------------------------搜索算法
# Be sure to first run `pip install hyperopt`
"""
from ray.tune.suggest.hyperopt import HyperOptSearch
def objective(x, a, b):
    return a * (x ** 0.5) + b


def trainable(config):
    # config (dict): A dict of hyperparameters.
    for x in range(20):
        score = objective(x, config["a"], config["b"])
        tune.report(score=score)  # This sends the score to Tune.
# Create a HyperOpt search space
config = {
    "a": tune.uniform(0, 1),
    "b": tune.uniform(0, 20)
}
#
# Specify the search space and maximize score
hyperopt = HyperOptSearch(metric="score", mode="max")

# Execute 20 trials using HyperOpt and stop after 20 iterations
tune.run(
    trainable,
    config=config,
    search_alg=hyperopt,
    num_samples=20,
    stop={"training_iteration": 20}
)
"""
# ------------------------------------------------------------------------------------------搜索算法
"""
from ray.tune.schedulers import HyperBandScheduler
def objective(x, a, b):
    return a * (x ** 0.5) + b


def trainable(config):
    # config (dict): A dict of hyperparameters.
    for x in range(20):
        score = objective(x, config["a"], config["b"])
        tune.report(score=score)  # This sends the score to Tune.

# Create HyperBand scheduler and maximize score
hyperband = HyperBandScheduler(metric="score", mode="max")

# Execute 20 trials using HyperBand using a search space
configs = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 1)}

tune.run(
    trainable,
    config=configs,
    num_samples=20,
    scheduler=hyperband
)

analysis = tune.run(trainable, search_alg=algo, stop={"training_iteration": 20})

best_trial = analysis.best_trial  # Get best trial
best_config = analysis.best_config  # Get best trial's hyperparameters
best_logdir = analysis.best_logdir  # Get best trial's logdir
best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
best_result = analysis.best_result  # Get best trial's last results
best_result_df = analysis.best_result_df  # Get best result as pandas dataframe
"""
# ------------------------------------------------------------------------------------------配置
"""
tune.run(trainable, num_samples=10, resources_per_trial={"cpu": 2})
"""
# ------------------------------------------------------------------------------------------限制并发
"""
algo = BayesOptSearch(utility_kwargs={
    "kind": "ucb",
    "kappa": 2.5,
    "xi": 0.0
})
algo = ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = AsyncHyperBandScheduler()
"""
# ------------------------------------------------------------------------------------------大型数据
"""
from ray import tune

import numpy as np

def f(config, data=None):
    pass
    # use data

data = np.random.random(size=100000000)

tune.run(tune.with_parameters(f, data=data))
"""



