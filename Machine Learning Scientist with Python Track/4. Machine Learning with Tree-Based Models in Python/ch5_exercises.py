# Exercise_1 
# Define params_dt
params_dt = {
'max_depth': [2, 3, 4],
'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]
}

--------------------------------------------------
# Exercise_2 
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)

--------------------------------------------------
# Exercise_3 
# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_


# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))

--------------------------------------------------
# Exercise_4 
# Define the dictionary 'params_rf'
params_rf = {
    
    'n_estimators' : [100,350,500],
    'max_features': ['log2','auto','sqrt'],
    'min_samples_leaf': [2, 10 ,30]
    
}

--------------------------------------------------
# Exercise_5 
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)

--------------------------------------------------
# Exercise_6 
# Import mean_squared_error from sklearn.metrics as MSE 
from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 

--------------------------------------------------
