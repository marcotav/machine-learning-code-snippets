# Machine Learning Code Snippets 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image title](https://img.shields.io/badge/work-in%20progress-blue.svg) ![image title](https://img.shields.io/badge/statsmodels-v0.8.0-blue.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/seaborn-v0.8.1-yellow.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/numpy-1.14.2-green.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg)
<br>

<br>
<br>
<p align="center">
  <img src="machine.jpeg" 
       width="350">
</p>
<br>


<p align="center">
  <a href="#cwa"> (1) Optimization of machine learning classification algorithms </a> •
  <a href="#comb"> (2) Selecting the model input features trying all possible combinations </a>
</p>

<a id = 'cwa'></a>
## 1) Optimization of machine learning classification algorithms

For this item I will use the `digits` dataset, which is included in scikit-learn.

### 1.1) Making imports and loading digits dataset

```
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # see the value of multiple statements at once.
pd.set_option('display.max_columns', None)
import pylab
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
import seaborn as sns
import math
import itertools
```
Now loading the dataset:
```
from sklearn.datasets import load_digits
digits = load_digits()
X,y = digits.data, digits.target
```

### 1.2) Train/test split

```
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                            test_size=0.25, random_state=0)
```

### 1.2) Comparing Models using the `f1-score`

To compare the models we will choose as our metric the `f1_score`. We can [write](https://www.scipy-lectures.org/packages/scikit-learn/index.html#supervised-learning-classification-of-handwritten-digits) a `for` loop that does the following:
- Iterates over a list of models, in this case, ` GaussianNB`, `KNeighborsClassifier` and `LinearSVC`
- Trains each model using the training dataset `X_train` and `y_train`
- Predicts the target using the test features `X_test`
- Calculates the `f1_score` comparing with `y_test`
- Note that the hyperparameters used for the three estimators are the default values

Note that we are iterating over a list of classes and the `( )` is inside the loop. The first loop is:

    LogisticRegression().fit(X_train, y_train)
    
an so on. Also using `.__name__` transforms the `LogisticRegression` into a string:

    type(LogisticRegression.__name__) --> str

```
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
def f1_score_comparison(models,X_train,X_test,y_train,y_test):
    for model in models:
        clf = model().fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = metrics.f1_score(y_test, y_pred, average="macro")
        print('%s: %s' % (model.__name__, round(f1,3)))
    return
    
models_list = [LogisticRegression, GaussianNB, KNeighborsClassifier, LinearSVC]

f1_score_comparison(models_list,X_train,X_test,y_train,y_test)
```

<br>
<br>
<p align="center">
  <img src="f1_score.png" 
       width="250">
</p>
<br>



We see that for the digits dataset and using the `f1_score` as metric, `KNeighborsClassifier` was the best classifier

### 1.3) Comparison using cross-validation

The cross-validation procedure is nicely illustrated below:

<br>
<br>
<p align="center">
  <img src="cross_val_concept.png" 
       width="500">
</p>
<br>

```
from sklearn.model_selection import cross_val_score
lst_av_cross_val_scores = []
for model in models:
    clf = model()
    cross_val_scores = (model.__name__, cross_val_score(clf, X, y, cv=5))
    av_cross_val_scores = list(cross_val_scores)[1].mean()
    lst_av_cross_val_scores.append(av_cross_val_scores)

model_names = [model.__name__ for model in models]

df = pd.DataFrame(list(zip(model_names, lst_av_cross_val_scores)))
df.columns = ['Model','Average Cross-Validation']

df.set_index('Model', inplace=True)

ax = df.plot(kind='bar', title ="Model Performance via Cross-Validation", figsize=(8, 8), fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Cross-Validation Score", fontsize=8);
plt.show()
```

<br>
<br>
<p align="center">
  <img src="plot_model_comp.png" 
       width="600">
</p>
<br>




Putting this into a function:
```
def cross_val_comparison(models,X_train,X_test,y_train,y_test):
    
    lst_av_cross_val_scores = []

    for model in models:
        clf = model()
        cross_val_scores = (model.__name__, cross_val_score(clf, X, y, cv=5))
        av_cross_val_scores = list(cross_val_scores)[1].mean()
        lst_av_cross_val_scores.append(av_cross_val_scores)
        model_names = [model.__name__ for model in models]

        df = pd.DataFrame(list(zip(model_names, lst_av_cross_val_scores)))
        df.columns = ['Model','Average Cross-Validation']
        df.set_index('Model', inplace=True)

    return df
```




<br>
<br>
<p align="center">
  <img src="cross_val_scores.png" 
       width="250">
</p>
<br>





We see that for the digits dataset and now using the cross-validation as metric, `KNeighborsClassifier` still is the best classifier.

We can write the former function with the same structure as the latter:

```
def f1_score_comparison(models,X_train,X_test,y_train,y_test):
    
    lst_f1 = []
    
    for model in models:
        clf = model().fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = metrics.f1_score(y_test, y_pred, average="macro")
        lst_f1.append(f1)
        model_names = [model.__name__ for model in models]
        
        df = pd.DataFrame(list(zip(model_names, lst_f1)))
        df.columns = ['Model','f1']
        df.set_index('Model', inplace=True)
        
    return df
```

<a id = 'comb'></a>
## 2) Selecting the model input features trying all possible combinations

Let us consider now another dataset, the Boston housing data. 

```
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
```

<br>
<br>
<p align="center">
  <img src="dfboston.png" 
       width="600">
</p>
<br>


The following function measure `r2` for all possible combinations of `features` and returns the best one:

```
def best_features(X,y,test_size,models,features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    combs = []
    for num in range(1,len(features)+1):
        lst = [list(i) for i in list(itertools.combinations(features, num))]
        combs.append(lst)
    r2_comb_lst = []
    best_comb = [0,[],' ']
    for lst in combs:  
        for m in models:
            for comb in lst:
                model = m().fit(X_train[comb],y_train)
                predicted = model.predict(X_test[comb])
                r2 = metrics.r2_score(y_test,predicted)**0.5
                r2_comb_lst.append([r2,comb,m.__name__])
                if r2 > best_comb[0]:
                    best_comb = [round(r2,4),comb,m.__name__]
    return 'Best combination is R2, features and model:',best_comb 
```
Calling the function as:
```
best_features(X,y,0.3,
               [LinearRegression,RidgeCV,LassoCV],
              ['CRIM', 'RM', 'B', 'LSTAT'])
```
we get:
```
('Best combination is R2, features and model:',
 [0.7699, ['CRIM', 'RM', 'LSTAT'], 'LassoCV'])
```

### To be continued


