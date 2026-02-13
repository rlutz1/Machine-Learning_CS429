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

### P4.py 

## Helper Files

All of these files can be found within the helper_code directory. They are to assist in keeping the project clean and separable. The following sections briefly describe each of their functions.

### Absorbed_Adaline.py

This contains the modifications made to absorb the bias into the weight vector for the Adaline Model. The evaluation of `net_input` corresponds with the proof given in the report, Section 1, to show validity.

### Absorbed_LogReg.py

This contains the modifications made to absorb the bias into the weight vector for the Logistic Regression model. The evaluation of `net_input` corresponds with the proof given in the report, Section 1, to show validity.

### Plotters.py

This contains helper methods for developing figures and plots for the report. For development use only.

### /unaltered_original_code/ Directory

This directory contains the original model code from the Raschka, et al. textbook.

+ Adaline.py: orginal Adaline model code. 
+ LogReg.py: orginal Logistic Regression model code. 
+ Perceptron.py: original Perceptron model code.