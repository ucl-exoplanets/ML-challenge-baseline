# ML-challenge-baseline
Repository hosting the baseline solution for [Ariel data challenge 2021](https://www.ariel-datachallenge.space/).

This code makes use of [numpy](https://github.com/numpy/numpy) and [pytorch](https://github.com/pytorch/pytorch).

## Repository content

- ```download_data.sh``` contains a shell script to download and extract the challenge's data
- ```utils.py``` contains different classes and functions:
    - ```ArielMLDataset``` to access the [data](https://www.ariel-datachallenge.space/ML/documentation/data) and feed it to our ML pipeline,
    - ```simple_transform``` to preprocess the data,
    - ```ChallengeMetric``` to define a scoring criterion following the general formula in the [documentation](https://www.ariel-datachallenge.space/ML/documentation/scoring) ,
    - ```Baseline``` which inherit ```torch.nn.Module``` class and defines the baseline solution.

- ```walkthrough.ipynb``` is a jupyter notebook showing the different steps of the baseline training and evaluation.

- Alternatively, ```train_baseline.py``` gives an example of a script to train the baseline model.

## Baseline solution

A feedforward neural network is trained on a subset of training examples selected randomly, while monitoring a validation score on some other random light curves. The neural network uses all 55 noisy light curves to predict the 55 relative radii directly. However, it does not use the stellar parameters, nor does it use any additional physics knowledge or ML technique to do so.

History:
- A first score of 9505 was recorded on April 12 using only 512 training samples and 512 validation samples.
- April 26th: A typo was identified in the preprocessing function during the cropping stage (many thanks to the participant who spotted it!). The preprocessing has been simplified, ignoring any cropping or ramp correction. In addition to this change, the validation set was set to 1024, the optimiser's learning rate to 0.0005 and a random seed (0) was set for repeatability in the dataset shuffling. The corresponding [new score is 9617](https://www.ariel-datachallenge.space/ML/leaderboard/).

## Preprocessing

The noisy light curves undergo the following preprocessing steps:

- i) 1 is removed to all light curve to have the asymptotic stellar flux centred around 0.
- ii) values are rescaled by dividing by 0.04 for the standard deviation to be closer to unity.

## Model & training hyperparaneters

We used a fully connected feedforward neural network of 2 hidden layers, made of 1024 and 256 units and both followed by Relu activation functions. The dimensions of the input and output layers are constrained by the input and output format, thus with 300*55 units and 55 units respecively.

The model was trained by minimizing the average MSE across all wavelengths using the ADAM optimizer with a learning rate of 0.0005 and beta parameters of (0.9, 0.999) (Pytorch defaults values). Training is performed until clearly overfitting, and the parameters associated with the highest validation score sare saved for later evaluation.

## What this baseline does not
None of these have been performed to produce thie baseline, leaving several potential ways for improvement:
- use of the full available dataset
- careful architecture comparison and choice
- test of various loss functions 
- use of regularisation methods
- hyperparmers optimisation
- inclusion of additional physical parameters (e.g. the ones given in input)
- data augmentation
- comparison with classical techniques

Although the data challenge has changed, various ideas to improve this model may be found in the [paper from 2019's ECML competition](https://arxiv.org/abs/2010.15996).

## Contact
For any question regarding the baseline solution: mario.morvan.18@ucl.ac.uk
