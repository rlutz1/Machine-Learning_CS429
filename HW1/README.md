# CS429 Assignment 1
Assembled by: Roxanne Krause, Kurukulasuriya Leitan, Marnina Willard
Submission Date: February 13, 2026

## Main Assignment Files 

All of the following files can be run through a simple `python .../Pk.py` command (or however it is preferred to be run), where k is the problem number. These contain the primary drivers of the implementation portion of the assignment.

### P1.py 

On run, this will request simply the Setosa/Versicolor classes from the Iris dataset and pulls, by default, sepal and petal length. Then, it trains the modified absorbed-bias Adaline, Logistic Regression models, followed by training the original Adaline and Logistic Regression models without the bias absorption. 

Then, it will simply print out the weights and bias of the models to confirm that the values are the same.

See `helper_code/Absorbed_Adaline.py` and `helper_code/Absorbed_LogReg.py` for the modifications made to absorb the bias.

### P2.py 

On run, this will default to grabbing Iris Setosa/Versicolor data, specifically sepal and petal length. Then, it will default to grabbing, wine Class 1/Class 2 data, specifically hue and color intesity.

Then, we will do the same for both Iris and wine data:
+ Set the learning rate `e_iris` and `e_wine` and epochs `i_iris` and `i_wine` for both Adaline and Logistic Regression models.
+ Train the two models with the above mentioned features and classes.
+ Plot the decision boundary between the classses for both models.
+ Plot the loss convergence for both models.

When running, 4 figures will appear, one after the other, the next appearing when the current figure is closed.

+ Figure 1: Decision boundary between Iris classes for both models.
+ Figure 2: Loss convergence for Iris training for both models.
+ Figure 3: Decision boundary between wine classes for both models.
+ Figure 4: Loss convergence for wine training for both models.

### P3.py 

On run, this will default to grabbing Iris Setosa/Versicolor/Virginica data, specifically sepal and petal length. All models will run with learning rate `e` and epochs `i` as set in script.

This driver will set up the One-Vs-All approach for multiclass classifications. It will use all 3 models (Perceptron/Adaline/Logistic Regression) for comparison. Then, we use a simple custom `TriClassPredictor` with a method `predict()` that will perform the choosing max `net_input` functionality explained in more detail in the report, Section 3.

Then we will plot the decision boundaries between all 3 classes of iris based on predictions made by the `TriClassPredictor`. Further, we will print out the proportion of accurate classifications made by the `TriClassPredictor` model.

### P4.py 

This driver begins by defining the needed methods to accomplish Stochastic Gradient Descent (SGD) and Mini-batch SGD. The methods are `fit_sgd` and `fit_mini_batch_sgd` respectively. These are then defined within the absorbed-bias Logistic Regression model.

Then, on run, we will run all three methods with Class 1 and 2 from the wine dataset and all features: Full Batch Gradient Descent, SGD, and Mini Batch SGD.

The figure this produces visuals on loss convergence, collecting the time it took to train, and final loss for each method.

## Helper Files

All of these files can be found within the helper_code directory. They are to assist in keeping the project clean and separable. The following sections briefly describe each of their functions.

### Absorbed_Adaline.py

This contains the modifications made to absorb the bias into the weight vector for the Adaline Model. The evaluation of `net_input` corresponds with the proof given in the report, Section 1, to show validity.

### Absorbed_LogReg.py

This contains the modifications made to absorb the bias into the weight vector for the Logistic Regression model. The evaluation of `net_input` corresponds with the proof given in the report, Section 1, to show validity.

### Plotters.py

This contains helper methods for developing figures and plots for the report that are used more than once. For development use only.

### /unaltered_original_code/ Directory

This directory contains the original model code from the Raschka, et al. textbook.

+ Adaline.py: orginal Adaline model code. 
+ LogReg.py: orginal Logistic Regression model code. 
+ Perceptron.py: original Perceptron model code.