# FairnessChecking

# What is what
Everything (Algorithms 1, 2, and 4) can be found in the `algorithms` directory. The main files here are:
1. **main.py**: The file that runs everything. Hyperparameters can be tweaked using this file and the command line arguments specified in the file. For instance, if I want `T_inner = 50` and `T = 200` with `epsilon = 0.05`, I would run:
`python main.py --T_inner 50 --T 200 --epsilon 0.05`
The details/flags available are in the file. Paths to csv files/data are also all in that file.
1. **meta_algo.py**: Algorithm 1 in the paper/the outer loop. Everything else gets run from here.
1. **bayesian_oracle.py**: Algorithm 4 in the paper/the inner loop.
1. **lambda_best_response_param_parallel.py:** Algorithm 2 in the paper/the LPs. I should probably rename this, but the ''param_parallel' part is just because I had a bunch of different versions of this and this is the parallelized one I settled on.

Can be safely ignored/helper files:
1. *evaluate_fairness.ipynb*: just helps me make some pretty graphs.
1. *lp_test.py*: just a file I used to check the LPs were working and how fast they were.
1. *voting_classifier.py*: the helper file that contains the VotingClassifier class I use to wrap multiple hypotheses into a majority vote classifier. this is called in the functions above, but no need to look into it.


