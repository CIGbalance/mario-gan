# Mario GAN
Mario level generation as an optimisation problem within the 
[GBEA benchmark](http://www.gm.fh-koeln.de/~naujoks/gbea/gamesbench_doc.html#mariogan) 
(detailed information about the optimization problems can be found there).

The data here is in compiled format (jar), while the source code is released in another 
Github repository: 
[https://github.com/TheHedgeify/DagstuhlGAN/tree/v1.0](https://github.com/TheHedgeify/DagstuhlGAN/tree/v1.0)

## Requirements

````
future>=0.17.1 
scipy>=1.3.1 
torch>=1.2.0 
torchvision>=0.4.0
````

(Earlier versions of these packages might also work.)

For `PyTorch`, follow the installation instructions [here](https://pytorch.org/).

## Test

Call
````
python mario_gan_evaluator.py 
````

to run a test evaluation for some selected Mario GAN functions.

Note that this does test the correctness of the evaluation, but rather checks that the 
code is being executed without problems.

## Usage in [COCO](https://github.com/ttusar/coco/tree/gbea)

Running the algorithm of your choice on one of the two Mario GAN test suites in COCO 
(the single-objective `rw-mario-gan` an the bi-objective `rw-mario-gan-biobj`) can range 
from trivial to not-so-trivial, depending on the language used. See the explanation 
[here](https://github.com/ttusar/coco/tree/gbea/code-experiments/rw-problems/GBEA.md).