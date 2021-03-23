# Midterm Project

### Question 1:	Import the weatherHistory.csv into a data frame. How many observations do we have? 
```markdown
import pandas as pd

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
df.shape
```
Using the above code allows us to see the number of observations and number of features in the dataframe. This code allows us to see that there are **96453.**


### Question 2:	In the weatherHistory.csv data how many features are just nominal variables?
```markdown
df.head
```
Based on this dataframe, there are **3** nominal variables.


### Question 3:	If we want to use all the unstandardized observations for 'Temperature (C)' and predict the Humidity the resulting root mean squared error is (just copy the first 4 decimal places):
```markdown
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
y = df['Humidity'].values
X = df['Temperature (C)'].values

model = LinearRegression()
model.fit(X.reshape(-1,1),y)

y_pred = model.predict(X.reshape(-1,1))
rmse = np.sqrt(mean_squared_error(y,y_pred))
rmse
 ```
As a result, the root mean squared error is **0.1514.**

### Question 4:	If the input feature is the Temperature and the target is the Humidity and we consider 20-fold cross validations with random_state=2020, the Ridge model with alpha =0.1 and standardize the input train and the input test data. The average RMSE on the test sets is (provide your answer with the first 6 decimal places):
```markdown
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
kf = KFold(n_splits=20, random_state=2020,shuffle=True)

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
y = df['Humidity'].values
X = df['Temperature (C)'].values

scale = StandardScaler()
model = Ridge(alpha=0.1)

PE = []
PE_train = []
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    X_train_scaled = scale.fit_transform(X_train.reshape(-1,1))
    y_train = y[train_index]
    X_test = X[test_index]
    X_test_scaled = scale.transform(X_test.reshape(-1,1))
    y_test = y[test_index]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))
```
The root mean squared error is **0.151438.**

### Question 5:	Suppose we want to use Random Forrest with 100 trees and max_depth=50 to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-cross validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 6 decimal places):
```markdown
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=1693,shuffle=True)

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
y = df['Humidity'].values
X = df['Apparent Temperature (C)'].values

model = RandomForestRegressor(n_estimators=100,max_depth=50)

PE = []
PE_train = []
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    model.fit(X_train.reshape(-1,1), y_train)
    y_pred = model.predict(X_test.reshape(-1,1))
    y_pred_train = model.predict(X_train.reshape(-1,1))
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))
```
The RMSE is **0.143483.**

### Question 6:	Suppose we want use polynomial features of degree 6 and we want to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 5 decimal places):
```markdown
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=1693,shuffle=True)

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
y = df['Humidity'].values
X = df['Apparent Temperature (C)'].values

polynomial_features = PolynomialFeatures(degree=6)
X_poly = polynomial_features.fit_transform(X.reshape((-1,1)))
model = LinearRegression()

PE = []
PE_train = []
for train_index, test_index in kf.split(X_poly):
    X_train = X_poly[train_index]
    y_train = y[train_index]
    X_test = X_poly[test_index]
    y_test = y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))
```
The RMSE is **0.14346.**

### Question 7:	If the input feature is the Temperature and the target is the Humidity and we consider 10-fold cross validations with random_state=1234, the Ridge model with alpha =0.2. Inside the cross-validation loop standardize the input data. The average RMSE on the test sets is (provide your answer with the first 4 decimal places):
```markdown
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=1234,shuffle=True)

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
y = df['Humidity'].values
X = df['Temperature (C)'].values

model = Ridge(alpha=0.2)
pipe = Pipeline([('scale', scale), ('Regressor', model)])

def DoKFold(X,y,model):
  PE = []
  for idxtrain,idxtest in kf.split(X):
    Xtrain = X[idxtrain]
    Xtest = X[idxtest]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(np.sqrt(MSE(ytest,yhat)))
  return np.mean(PE)
  
DoKFold(X.reshape(-1,1),y,model)
```
The RMSE is **0.1514.**

### Question 8:	Suppose we use polynomial features of degree 6 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 4 decimal places):
```markdown
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=1234,shuffle=True)

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
y = df['Temperature (C)'].values
x1 = df.loc[:, df.columns != 'Temperature (C)']
X = x1.drop(['Formatted Date', 'Summary', 'Precip Type', 'Apparent Temperature (C)', 'Visibility (km)', 'Loud Cover', 'Daily Summary'],axis=1)

polynomial_features = PolynomialFeatures(degree=6)
X_poly = polynomial_features.fit_transform(X)
model = LinearRegression()

PE = []
PE_train = []
for train_index, test_index in kf.split(X_poly):
    X_train = X_poly[train_index]
    y_train = y[train_index]
    X_test = X_poly[test_index]
    y_test = y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))
```
The RMSE is **6.1210.**

### Question 9:	Suppose we use Random Forest with 100 trees and max_depth=50 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 4 decimal places):
```markdown
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=1234,shuffle=True)

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
y = df['Temperature (C)'].values
x1 = df.loc[:, df.columns != 'Temperature (C)']
X = x1.drop(['Formatted Date', 'Summary', 'Precip Type', 'Apparent Temperature (C)', 'Visibility (km)', 'Loud Cover', 'Daily Summary'],axis=1).values

model = RandomForestRegressor(n_estimators=100,max_depth=50)

PE = []
PE_train = []
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    PE_train.append(np.sqrt(MSE(y_train,y_pred_train)))
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.mean(PE_train)))
print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.mean(PE)))
```
The RMSE is **5.8338.**

### Question 10:	If we visualize a scatter plot for Temperature (on the horizontal axis) vs Humidity (on the vertical axis) the overall trend seems to be 
```markdown
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('drive/MyDrive/Colab Notebooks/weatherHistory.csv')
y = df['Humidity'].values
X = df['Temperature (C)'].values

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X,y,color='blue',s=40)
ax.set_xlim(-20, 50)
ax.set_ylim(0, 1)
ax.set_xlabel('Temperature',fontsize=14,color='navy')
ax.set_ylabel('Humidity',fontsize=14,color='navy')
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
plt.tick_params(axis='x', colors='navy')
plt.tick_params(axis='y', colors='navy')
ax.minorticks_on()
plt.show()
```
This graph shows a **decreasing** trend.


# Midterm Questions
### Question 1:	Match each of the below data types to it's description. Note there are some answers that will not be matched.

Ordinal data has unique indentifiers and order matters. Discrete data describes nominal data. Continous data deals with numeric values which can be interpreted further. 

As a result, **discrete data** is data which represents specific classes without order (i.e., USA and France) ; **ordinal data** is data in which a larger number indicates a larger amount, but not the absolute amount (i.e., a ranking of the top ten colleges in the U.S.A.) ; **continuous data** is data in which larger values indicate a larger absolute amount (i.e., the cost of ten different coffee options at the daily grind).

### Question 2:	What is true about feature scaling? Choose all that apply. Incorrect choices will result in negative points.

It is **true** that there are many different approaches to feature scaling including z-scores, 0 to 1 scaling, Min Max Scaler, etc. The goal of feature scaling is to combine features in some way so that calculations, such as the Euclidean distance, can be calculated. 

On the other hand, K-Fold cross validations are useful for understanding the predictive powers of various models; they are not an example of feature scaling. Likewise, density estimates are not examples of feature scaling. Also, feature scaling is completed *before* model results are found, not after.


### Question 3:	Do you agree or disagree with the following statement:	In a linear regression model, all variables must be normally distributed for the model to fit.

This is **false.** Linear regression does not assume that all variables must be normally distributed to fit. Instead, the **residuals** of the model should have a normal distribution.


### Question 4:	How many iterations of a random forest should you run?

Based on our lectures, we should allow the trees 'to grow' so typically the more, the better. However, there is no good pre-determined number for the trees. Our end goal is to chose a number that allows for convergence (when adding more trees doesn't matter to your results) **but, this number heavily depends on the structure and complication of your study.** In other words, **it depends on the nature of your data and study.**


### Question 5:	In SVR, what does the epsilon parameter define?

Support Vector Regression is used to solve a quadratic optimization problem with specific contraints. Based on these constraints, *epsilon* is used to determine the width above and below the optimization. This definition, and example graphs from lecture, demonstrate that epsilon is **the width of the zone within which you will not account for errors.**


### Question 6:	Regression trees are able to model non-linear systems.

It is important to understand that non-linear relationships exist in the world, so we need to be able to accommodate them. Regression trees are able to model these possibly non-linear data sets. For example, the lectures provided the example of the use of regression trees on the *possibly non-linear* relationship for the price of a car based on the combination of specific features. As a result, this question is **true.**


### Question 7:	What is ensemble learning (such as in Random Forests)?

The word *ensemble* represents a group of people, in machine learning, this idea stays true. Ensemble learning is **the use of multiple models to identify a solution to a problem.** Examples include both Decision Trees and Random Forests.


### Question 8:	For support vector regression slack variables are

According to our lectures, slack variables are used to accommodate points that are 'close' to the epsilon margins and that may influence the value of the weights. Comparing the equations for SVR before and after slack variables allow us to understand that they are **in absolute value added as a penalty to the minimization of the sum of squared residuals.**


### Question 9:	The root node of a classification tree is...

Based on lecture, a root node contains all the data points. Likewise, DATA 146 stated that the root node **"contains all of the data."**


### Question 10:	What is the Posterior Probability in a Bayesian approach to classification?

The Posterior Probability is the end result we are trying to find using probability formulas according to the Bayesian approach. The other aspects of the Bayesian approach include the likelihood, prior, and marginal. However, these are inputs into the formula while the **posterior probability is the output we are attempting to find - i.e. the probability you are solving for, for each class.**


### Question 11:	What are key benefits of classification forests, as contrasted to classification trees?

Classification forests are a combination of many classification trees. According to our lectures, classification trees are "more resilient to outliers, better for external validity and you can also provide information on how certain or uncertain you are about a result." This description allows us to understand that **by running multiple instances of the same model, the risk of overfitting to outliers is reduced.**


### Question 12:	In the context of a loss function, what does a L1 Norm calculate?

The L1 norm minimizes the sum of the absolute differences between the true and predicted values. This makes **the summation of the absolute difference between each true and predicted value** the correct answer.


### Question 13:	In the context of a Loss function, what is true when contrasting a L1 norm to a L2 norm?

L1 norm minimizes the sum of the absolute differences between the true and predicted values. However, L2 norm minimizes the sum of squares differences between the true and predicted values. As a result, the errors for both of these norms are different. For example, if the difference true and predicted values was 2 for both, then the L1 norm would equal 2 while the L2 norm would equal 4. As a result, **A L1 norm accounts for error in a linear fashion, in which an estimate that is off by 1 is accounted as an error of "1", and an estimate that is off by 2 is accounted as an error of "2". Contrasting to this, a L2 norm penalizes for errors of larger magnitudes at a different rate compared to the L1 norm** is the correct answer.


### Question 14:	You can create an ensemble model by re-fitting the same model many times with randomly selected data.

This question is **true.** For example, random forests do this when they construct multiple different decision trees. As a result, random forests are considered *ensemble models.*


### Question 15:	Grouping data into quantiles is accomplished by subdividing data into intervals of equal width

This statement is **false.** According to our lectures, quantiles are defined as "a set of values for a random variable that divide its frequency distribution into groups, each containing the same fraction of the whole data." Based on this definition, it is evident that quantiles group data into areas with *equal probabilities* not equal widths.


### Question 16:	A system is linear in its parameters when...

One main aspect of linear models is that they must be linear in terms of their weights. Because of this, **coefficients do not appear as exponents** for linear models. Also, according to Lecture 9, y hat is really just the matrix-vector product between the feaures and the weights. Based on this definition, it is true that **the system can be written as the product of a feature matrix and a coefficient vector.** 

However, *none of the independent variables are raised to a power or multiplied by one another* is false because a system may still be linear even if an independent variable is raised to another power. Likewise, because of this previous statement, this also makes *the model describes a straight line* false too.


### Question 17:	In contrast to Ridge regression, LASSO regression...

It is **true** that Lasso is **can effectively remove features from your model by setting their coefficients to zero** because it facilitates variable selection. When comparing Ridge and Lasso regression, one key difference is that Lasso **uses the L1 norm of parameters as an additional penalty rather than the L2 norm.** This is true because Lasso uses the L1 norm while Ridge uses the L2 norm.

However, it is **false** that Lasso will *always be the better model to use* because one specific model will never *always* be the most certain choice. Also, both Ridge and Lasso use the *sum of squared residuals to fit a model* so this option is **false** too.


### Question 18:	Suppose you did an ordinary least squares regression, and a plot of the residuals vs. an independent variable looked like below:

This graph has a near-zero mean. Likewise, based on its shape, there is a normal distribution of residuals which does not make it less reliable in certain regions. Finally, this graph would not allow us to understand if there is multicollinearity or not. 


### Question 19:	If you are presented a data set with multiple features and some of the observations have missing values, what is the correct order of steps to take if you would like to build a regression model?

When creating a regression model, the first thing to always do is to **import the data set.** After that, you need to 'get to know' the data, meaning that if there are missing values, you must **impute or drop the missing values.** Next, you must **subdivide the data into 'Train' and 'Test' sets.** Then, **scaling the features** is the next step to take. Scaling is done after subdividing the data because biases could potentially make its way into the dataset if it is done prior to the 'Train' and 'Test' split. **Deciding between a linear or polynomial model** is next so that the model can be created. Next, **fitting the regression model** is the appropriate step because the data has been split and the model has been defined. Once all of these previous steps have been completed, you may finally **compute the root mean square error for the "Test" set.**


### Question 20:	In a simple linear regression the goal is to

The goal of simple linear regression is to **estimate the expected value of the predicted variable.** When creating simple linear regressions, we are trying to *predict* a specific variable based on our feature(s). Doing this will allow us to draw conclusions about the relationship between the feature(s) and the target variable.


### Question 21:	Assume that data is standardized by z-scores and a linear regression model is fit with one feature (input) variable. If the residuals are not normally distributed and they seem to be increasing when the predicting variable is decreasing, choose among the following other regression models the ones that may help improve the predictions.

According to the question, the residuals *are not normally distributed* - to use linear regression, residuals **must** follow a normal distribution. Both Ridge and Lasso regressions are linear regression models, so those would not be useful. SVM is also used for linear regression, so it also would not be useful. This leaves us with **a nonlinear regression algorithm such as Decision Trees or Random Forests**. 
