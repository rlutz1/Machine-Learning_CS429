# CS429 Assignment 2
Assembled by: Roxanne Krause, Kurukulasuriya Leitan, Marnina Willard \
Submission Date: February 27, 2026

## Main Assignment Files 

All of the following files can be run through a simple `python .../Pk.py` command (or however it is preferred to be run), where k is the problem number. These contain the primary drivers of the implementation portion of the assignment.

### P1.py 

### P2.py 

On run, this file will plot a demo figure of randomly generated linearly separable data with the following parameters by default:
`d = 2, n = 100, u = 100, rand_seed = 42`

This also controls the generation of files based on the data generated. By default, files will not be created, but is controlled by top level constant boolean `GEN_TO_FILE`.

### P3.py 

### P4.py 

## Helper Files

All of these files can be found within the `helper_code` directory. They are to assist in keeping the project clean and separable. The following sections briefly describe each of their functions.

### DatasetGenerator.py

This is a simple helper file with two functions: write a given dataset with true labels to 2 files (70% to _TRAIN, 30% to _TEST), and then read it back to a useable Python list. The files are simply a CSV format in which each row is a data sample, with the very last entry being its true label (1/-1). 

This is used primarily in P2, but the use will be turned off for the final submission so as to not generate files when running every time.

## Datasets

This contains the data set CSV's used in P3 and P4. As mentioned, the files are simply a CSV format in which each row is a data sample, with the very last entry being its true label (1/-1).

The file names indicate the `d` and `n` parameters used for each dataset: `d{number dimensions}n{number samples}_{TEST or TRAIN}.csv`. All sets were generated with the following additional parameters: `u = 100, rand_seed = 42`.
