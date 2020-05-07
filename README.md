# Mario GAN
Mario level generation as an optimisation problem within the [GBEA benchmark](http://www.gm.fh-koeln.de/~naujoks/gbea/gamesbench_doc.html#mariogan) (detailed information about the optimization problems can be found there).

The data here is in compiled format (jar), while the source code is released in another Github repository: [https://github.com/TheHedgeify/DagstuhlGAN/tree/v1.0](https://github.com/TheHedgeify/DagstuhlGAN/tree/v1.0)

## Requirements

This code requires `PyTorch`, follow the installation instructions [here](https://pytorch.org/).

## Test

Call
````
python mario_gan_evaluator.py 
````

to run a test evaluation for some selected Mario GAN functions.

Note that this does test the correctness of the evaluation, but rather checks that the code is being executed without problems.
