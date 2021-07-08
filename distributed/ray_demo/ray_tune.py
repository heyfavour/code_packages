from ray import tune


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
