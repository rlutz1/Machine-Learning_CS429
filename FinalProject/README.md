# CS429 Final Project
Assembled by: Roxanne Krause, Kurukulasuriya Leitan, Marnina Willard \
Submission Date: May 8, 2026

## Main Assignment Files 

The details of the contents and optional running of these files are as follows.

### MapCompressor.py

This file contains the MapCompressor class that conducts the "over-approximation" of the given BMP file as described in the report. An extra Python library Pillow is used for reading in the BMP file.

No action will occur on this file run--it is only for use in other scripts.

### Environment.py

This file contains the Environment class that is used to define reward strategies and allow for agent interaction. It takes in the compressed map produced by MapCompressor and abstract the environment for agent training and testing.

No action will occur on this file run--it is only for use in other scripts.

### Agent.py 

This file contains a baseline Agent class that is used to enable a consistent start of the QLearn and SARSA agents. It covers the basic needs of the agent: initialize the Q Table, update the Q Table, interact with the environment, train, and test.

No action will occur on this file run--it is only for use in other scripts.

### QLearnAgent.py 

This file contains a QLearn Agent class that is used for evaluation. It overrides all Agent baseline methods with the needed Q-Learn rules and implements a training and testing iteration. 

On running this file, images given in the report with the parameters below are created and saved to the current working directory.
+  Learning Rate: 0.5
+  Epsilon: 0
+  Discount: 0.5
+  Episodes: 5000
+  Steps: 1000
+  Strategy: S1
+  Testing Start Point: (0, 0)
+  Testing End Point: Bottom right corner of the given map

### SarsaAgent.py 
This file contains a SARSA Agent class that is used for evaluation. It overrides all Agent baseline methods with the needed SARSA rules and implements a training and testing iteration. 

On running this file, images given in the report with the parameters below are created and saved to the current working directory.
+  Learning Rate: 0.5
+  Epsilon: 0
+  Discount: 0.5
+  Episodes: 5000
+  Steps: 1000
+  Strategy: S1
+  Testing Start Point: (0, 0)
+  Testing End Point: Bottom right corner of the given map

### Evaluate.py

This file aided in providing consistent metric gathering methods and specific experiments (as required by report and otherwise) to run with specified parameters. This enabled a clean testing environment for any given run.

Running this file will yield the running of (with default parameters according to the base requirements of the given assignment)
1. Map complexity experiment
2. Exploration rate experiment
3. Discount rate experiment
4. Reward strategy experiment*

*The first 3 of the above experiments will yield data collected directly to the report. The 4th will default to selecting exploration rate and discount bests in isolation, and run the reward strategy experiment. However, the reward strategy experiment was run specifically with the best parameters mentioned in the report to gather the final data in that section using this experiment method.

## Data directory

This directory contains the following.
+ **Original Images**: the given BMP images to use for the project.
+ **QLearn Images**: the images generated and depicted in the report.
+ **SARSA Images**: the images generated and depicted in the report.