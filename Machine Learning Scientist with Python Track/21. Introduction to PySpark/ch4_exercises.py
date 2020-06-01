# Exercise_1 
# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()

--------------------------------------------------
# Exercise_2 
# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
# the curve is the ROC, or receiver operating curve.

--------------------------------------------------
# Exercise_3 
# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0,1])

# Build the grid
grid = grid.build()

--------------------------------------------------
# Exercise_4 
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator)

--------------------------------------------------
# Exercise_5 
# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

--------------------------------------------------
# Exercise_6 
# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))

--------------------------------------------------
