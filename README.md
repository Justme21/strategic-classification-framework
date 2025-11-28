# Strategic Classification Framework

This repository contains the code to reproduce the experiments in

* J. Geary, B. Gao, H. Gouk. [Computing Strategic Responses to Non-Linear Classifiers](https://arxiv.org/abs/2511.21560) EurIPS 2025 Workshop: Unifying Perspectives on Learning Biases.

## Background

Strategic Classification considers interactions between an agent attempting to train a `classifier` model by minimising a `loss` when the targets being classified behave strategically according to their `best response`, which optimises the difference between the `utility` of a state change, and the `cost` associated with the state change. 

## Using This Repo

This is a modular framework for training classifier models in strategic settings. The modular units can be specified from the command line. New modules can be added to the appropriate subdirectories and the module class definitions can all be found in the [Clf_Learner/interfaces](https://github.com/Justme21/strategic-classification-framework/tree/main/Clf_Learner/interfaces) subdirectory. Arguments can be passed as a dict to each of the module units using the --args command line argument, and specifying the intended module.

Explanations for the remaining arguments can be found using --help.

Sample code demonstrating how to run the experiment code can be found in [run.sh](https://github.com/Justme21/strategic-classification-framework/blob/main/run.sh).

The notebooks to reproduce the datasets and the visualisations used in the experiments can be found in the `data_tools` directory.

