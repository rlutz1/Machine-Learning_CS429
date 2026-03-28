# CS429 Midterm Project
Assembled by: Roxanne Krause, Kurukulasuriya Leitan, Marnina Willard \
Submission Date: March 27, 2026

## Main Assignment Files 

To run any of the following scripts, you must be in the top level directory (`MidtermProject`). This is due to the path for the reading of the IDX fles of the data sets.

### MNISTReader.py, MNISTFashionReader.py

These files accomplish Tasks 1 and 2 within the `__init__` construction. We use Idx2Numpy to conver the files to Numpy arrays, followed by flattening them to one dimension with Numpy's `reshape()`. After that, all calls to grab the training and testing images and labels are as simple accessing the field, and they will be correctly formatted to use for all fittings.

### P3_*.py Files

These are all the files that were useful in accomplishing baseline and targeting runs. 

These files are essentially the same functionality, but with specific iteration and kernel information identifying the more important identifiers of the run. Within, we 

1. Set up all dimensionality reducers (PCA 50, 100, 200, and LDA).
2. Set up what combinations of parameters to run.
3. Set up the kernel to run with.
4. Set up a specific number of iterations.
5. Create a pipeline to transform the data with `StandardScaler`, reduce with a specific reduction method, and then fit the SVC model.
6. Score the model test and train accuracy once fitting is complete.
7. Write all collected time and accuracy data to a corresponding CSV.

The reason behind having many different scripts with different names is due to the fact that running on one machine quickly became problematic, and so the UNM B146 Computer Science Lab machines were used in parallel to run smaller data collections. This enabled us to get a more thorough view of parameter successes and failures at low iteration counts, highlighting trends and get a better feel for what works and what doesn't. Further, it became useful as well when targetting, since we could run many targets in parallel as well and gather more data.

All of these files together encompass the process for collecting data for Task 3.

### P3_visualize_*.py Files

The files `linear_baselines.py`, `rbf_baselines.py`, `poly_baselines.py`, are the primary used to generate visuals for clearly seeing potentially good parameter balances. These will create the accuracy plots included in the report, corresponding to the kernel in the file name.

These files pull the necessary entries from the data collection CSVs in the `data` directory for the corresponding data set and generate a plot with Python library MatPlotLib.

`dim-reducers.py` was a quick compilation generated when writing the report (and by no means good coding practice) to generate visuals of the comparisons of parameter combinations per kernel for the dimensionality reduction methods. This will generate all basic figures included in the PCA vs LDA section of the report.

### P4.py

This file contains the completion of Task 4, using bootstrap aggregating to compare against our best numbers from Task 3. It contains the class used to group the SVC models, evaluation, and actual fitting method.

## Helper Files

### TimeWrappers.py

This is the main way of collecting fine-grained timing information of scaling, transforming, and fitting. We include two custom wrapper classes:
1. `TimedTransform`
2. `TimedClassifier`

The classes act as a very basic wrapper on the transformers (scaler and dimension reduction) and classifier (SVC). Through some experimentation, the main timing information is pulled from the pipeline's call to the transformer's `fit_transform` and the classifier's `fit`. All specific times come from using Python's `time.perf_counter()` within these function wrappers. These wrappers are not intended and not recommended for continuous tracking--they are designed to primarily grab the timing from the pipeline calls, and that is all.

## Data

Within the P3 folder, there is an additional branching to separate out the MNIST and MNIST Fashion sets. Within both are three files:
1. `initial-fit-times.csv`
2. `test-train-acc.csv`
3. `final-times.csv` **unused in analysis**

`initial-fit-times.csv` contains the identifying information of a run, and specifically the scale, transform, and fit times grabbed by the timing wrappers post-pipeline-fit. `test-train-acc.csv` contains similar identifying run information, but with the test and training set accuracy score. As mentioned, `final-times.csv` is included, but was not used in final report writing or analysis due to the timing wrappers not being completely thorough in this collecion. 

All numbers in the report come from these files.

## Datasets

The `datasets` directory contains the downloaded IDX files for both MNIST and MNIST Fashion sets. 