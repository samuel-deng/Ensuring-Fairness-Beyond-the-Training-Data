# Ensuring Fairness Beyond the Training Data

## Preliminary Tests
`main/Part1_CheckFairness` just contains some preliminary tests we did on COMPAS and Adult to see how existing fair classifiers performed vs. standard LogisticRegression. These can be safely ignored. 

## Examining Marginal Distributions and Perturbations
Can be found in `main/Part1_Marginal`. In these files, we used existing fair classifiers to check how non-robust they were to weighted perturbations in the data. to visualize this, we plot the marginal distributions of each feature in the data, doing this for Adult and COMPAS. We find that the marginals are almost identical, but fairness is quickly violated. These experiments correspond to Figure 1 in the paper.

## Training Robust Classifier
Our robust classifier algorithms (Algorithms 1, 2, and 3) can be found in the `main/Part2_MetaAlgo_FTRL` directory. The main files here are:
1. **main.py**: The file that runs everything, and trains your robust classifier. It will output a single .pkl file that contains the robust classifier in the form of a VotingClassifier object. Hyperparameters can be tweaked using this file and the command line arguments specified in the file. For instance, if I want `T_inner = 50` and `T = 200` with `epsilon = 0.05`, I would run:
`python main.py --T_inner 50 --T 200 --epsilon 0.05`
The details/flags available are in the file. Paths to csv files/data are also all in that file.
1. **meta_algo.py**: Algorithm 1 in the paper/the outer loop. Everything else gets run from here.
1. **bayesian_oracle.py**: Algorithm 3 in the paper/the inner loop, AKA the ApxFair algorithm.
1. **lambda_best_response_param_parallel.py:** Algorithm 2 in the paper/the LPs. This is the best response from the Lambda player in the ApxFair algorithm.

Can be safely ignored/helper files:
1. *evaluate_fairness.ipynb*: just helps me make some pretty graphs.
1. *voting_classifier.py*: the helper file that contains the VotingClassifier class I use to wrap multiple hypotheses into a majority vote classifier. this is called in the functions above, but no need to look into it.

For the exact experiments that we ran in the paper (with tuned hyperparameters), run `final_DP_1000.sh` and `final_EO_500.sh`for training robust classifiers on all the train/test splits and all datasets.

## Evaluation
The evaluation and comparison of the robustness of our classifier can be done through `main/Part3_Comparisons`. There is a notebook for each dataset and fairness definition, the saved plots under `neurips-plots` and all of our trained models in `trained_robust.` The `ensemble_final` folder in `trained_robust` contains all the models we used for our experiments. The `h0_final` contains the models that are given only by running ApxFair (i.e. one iteration of the inner loop).

Because of file size requirements for subimssion to NeurIPS, we only include the trained models for the first train/test split. However, running `final_DP_1000.sh` and `final_EO_500.sh` will duplicate the exact conditions to train the models for all five train/test splits for each dataset and fairness definition.

## How To Run
Get off the ground by setting up the basic environment just using the included `environment.yml` file:
`conda env create -f environment.yml`
This should be sufficient to start running `main.py.` An example run:

`python main.py --T_inner 50 --T 200 --epsilon 0.05`

There are a lot more tweakable hyperparameters than can be changed through command line; just check the `main.py` file to see all of those flags. Every hyperparameter corresponds to the symbols used in the paper.
