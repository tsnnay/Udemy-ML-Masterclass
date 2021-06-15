# Theory and Intuition
First Hyperplanes and margins are discussed and incremental complexity in classifiers is shown, finally ending at Support Vector Machines

* Maximum Margin Classifier
* Support Vector Classifier 
* Support Vector Machines

## Hyperplanes And Margins
Understanding hyperplanes and margin classifiers in fundamental to understanding SVM

### What is Hyperplane?
In a N-dimensional space, a hyperplane is a flat affine subspace of hyperplane dimension N-1. For ex:
* For 1-D, Hyperplane is a single point
* For 2-D, Hyperplane is a line and so on

### Why we bother with hyperplanes?
The main idea is to use hyperplane to create separation between classes (fancy way of saying different types of data points). Our hope is such a hyperplane exists which can bifurcate differencing points in the space

### Baby Chick Example
1-D dataset of baby chicks (male or female) is plotted on a line, with maleo on RHS and female on LHS. Visually, data is easily separable.

I this dataset, a single point in the middle of dataset can easily separate both datasets, but issue is **placing of the point**

**Where to Place the Point?**
Many points exists which can perfectly separate the points. We use a separator that maxifies the margin between hyperplane and first instance of each class. Such an hyperplane which maximises the margins is called __Maximum Margin Classifier__

Same example can be applied in 2-D space, where classes are separated by line, but there are $\inf$ lines which can separate. Again, we use maximise the margin (distance) between line and data points (each data point 2D vector here)

### Unseparable Classes?
NOt all classes will be easily separable (at least visually) showin in Fig. A single point will mess up at least one class, so we need to chose our poison. Our decision is guided by Bias-Variance Tradeoff

**Bias Variance Tradeoff**
Example @fig:unbalanced_margin and @fig:skewed_margin_2d show how one can overfit hard to one particular dataset (however no. of point misclassified is same i.e. 1 in both cases). But it is obvious our maximum margin classifier skews heavily female in _ and in _. This can bite us in ass when we decide to test or deploy the model

![Variance Fit to Unbalanced Data \label{unbalanced_margin}](img\soft-margin=classifer1.PNG){#fig:unbalanced_margin}

This is called **high variance fit**. In example _, it was "picking too much noise from female data" and thus, overfitting it or had a high variance w.r.t female data points. We can introduce **bias** for more logical classifier, even at the cost of training accuracy. This misclassification doing margin is called a *_soft margin_*, which allows for miscallsification within itself. We manipulate soft margin with introduction of bias. 

**But again, there are multiple threshold splits of soft margin, and **Maximum Margin** concept is already applied, so what else we can do to get optimum *soft margin*.**
The answer lies in level of misclassification to be allowed. With *misclassification* as our north-star, we perform *Cross Validation* to figure out best *soft margin* amongst all.

### Soft Margin Demonstraion
Maximum margin classifer in this example skews heavily female, due to picking "too much variance or too much noie" from the female set. The highlighted figure _ shows Male classification zone is too larger than what seems necessary, and it can cause problems in test set. So, there is need to soften the margin we got from *Maximum Margin Classifer*

![Skewed Max. Margin Classifier 2D](img\skewed-soft-margin-2d.png){#fig:skewed_margin_2d}

So we introduce a new classifier, that allows for *soft margins*, called a *Support Vector Classifier*

**What happens when Hyperplane theory falls on it face?**
Cases _ and _ demonstrate the respective hyperplanes (point and lines resp.) fail here. Using multiple points can solve the issue in case _, but in _, group of lines is not gonna help much

*In general, higher dimesions make it difficult to use multiple hyperplanes*

And that's the limitation of *Support Vector Classifier* and rationale to move on to *Suppport Vector Machines*

![This is caption ](wine.jpg){#fig:1}

See @fig:1

