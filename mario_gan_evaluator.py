import sys
import subprocess


# This executes the GAN and evaluates it
# problem ids
# id = g + f*G + c*F*G
# where g is GAN-type (from which json)
# f is function
# c is whether concatenated or not
import torch
from torch.autograd import Variable

import sys
import os
import numpy
from .gan_implementation.models import dcgan
import glob
from collections import OrderedDict

batchSize = 64

imageSize = 32
ngf = 64
ngpu = 1
n_extra_layers = 0

features = 13
budget = 5000

GROUND = 0
BREAK = 1
PASS = 2
QUESTIONC = 3
QUESTIONP = 4
COIN = 5
TUBE = 6
PLANT = 7
BILL = 8
GOOMBA = 9
GKOOPA = 10
RKOOPA = 11
SPINY = 12

sim=1
path = os.path.dirname(os.path.abspath(__file__))

#        tiles.put('X', 0); //solid
#        tiles.put('x', 1); //breakable
#        tiles.put('-', 2); //passable
#        tiles.put('q', 3); //question with coin
#        tiles.put('Q', 4); //question with power up
#        tiles.put('o', 5); //coin
#        tiles.put('t', 6); //tube
#        tiles.put('p', 7); //piranha plant tube
#        tiles.put('b', 8); //bullet bill
#        tiles.put('g', 9); //goomba
#        tiles.put('k', 10); //green koopas + paratroopas
#        tiles.put('r', 11); //red koopas + paratroopas
#        tiles.put('s', 12); //spiny + winged spiny

batchSize = 1

#################################################################################
# Utils


def exist_gap(im):
    # gap exists if not 10 (coin), not 2 (passable)
    width = numpy.shape(im)[1]
    height = numpy.shape(im)[0]
    gaps = numpy.zeros(width)
    for i in range(0, width):
        imc = im[:, i]
        unique, counts = numpy.unique(imc, return_counts=True)
        dist = dict(zip(unique, counts))
        if dist.get(COIN, 0) + dist.get(PASS, 0) == height:  # all tiles in column passable
            gaps[i] = 1
    return gaps

def count_gaps(im):
    gaps = exist_gap(im)
    return sum(gaps)

def gap_lengths(im):
    gaps = exist_gap(im)
    gaps = "".join([str(int(x)) for x in gaps])
    return list(map(len, gaps.split('0')))

def max_gap(im):
    return max(gap_lengths(im))

def count_tile_type(im, tile):
    num_tiles = (len(im[im == tile]))
    return num_tiles

def gan_maximise_tile_type(x, nz, tile):
    return -count_tile_type(x, nz, tile)

def gan_target_tile_type(x, nz, tile, target):
    return abs(target - count_tile_type(x, nz, tile))

def gan_target_tile_type_frac(im, tile, target_frac):
    total_tiles = float(numpy.shape(im)[0] * numpy.shape(im)[1])
    return abs(target_frac - (count_tile_type(im, tile) / total_tiles))

def tilePositionSummaryStats(im, tiles):
    coords = numpy.where(im==tiles[0])
    if len(coords[0])==0:
        return 0, 0, 0, 0
    x_coords = coords[1]
    y_coords = coords[0]
    for tile in tiles[1:]:
        tmp = numpy.where(im==tile)
        x_coords = numpy.append(x_coords, tmp[1])
        y_coords = numpy.append(y_coords, tmp[0])
    return numpy.mean(x_coords), numpy.std(x_coords), numpy.mean(y_coords), numpy.std(y_coords)

def executeSimulation(x, netG, dim, fun, agent):
    java_output = subprocess.check_output('java -Djava.awt.headless=true -jar '+path+'/dist/MarioGAN.jar "' + str(x) +'" "' + netG + '" '+str(dim)+' '+str(fun)+' '+str(agent) +' ' +str(sim), shell=True);
    lines = java_output.split(b'\n')
    result = lines[11+sim].decode("utf-8")
    if "Result" not in result:
        raise ValueError('MarioGAN.jar output not formatted as expected, got {} '.format(result))
    return float(result[6:])


################################################################################################
# Fitness Functions

# Estimates the leniency of the level
# Value range?
# minimise
def leniency(x, netG, dim):
    im = translateLatentVector(x, netG, dim)
    unique, counts = numpy.unique(im, return_counts=True)
    dist = dict(zip(unique, counts))
    val = 0
    val += dist.get(QUESTIONP, 0) * 1
    val += dist.get(PLANT, 0) * (-1)
    val += dist.get(BILL, 0) * (-1)
    val += dist.get(GOOMBA, 0) * (-1)
    val += dist.get(GKOOPA, 0) * (-1)
    val += dist.get(RKOOPA, 0) * (-1)
    val += dist.get(SPINY, 0) * (-1)
    val += count_gaps(im) * (-0.5)
    t = numpy.array(gap_lengths(im))
    if count_gaps(im) > 0:
        val -= numpy.mean(t[t != 0])
    return val

# Percentage of stackable items
# Value range 0-1
# maximise
def density(x, netG, dim):
    im = translateLatentVector(x, netG, dim)
    unique, counts = numpy.unique(im, return_counts=True)
    dist = dict(zip(unique, counts))
    width = numpy.shape(im)[1]
    height = numpy.shape(im)[0]
    val = 0
    # Tiles you can stand on
    val += dist.get(GROUND, 0)
    val += dist.get(BREAK, 0)
    val = float(val) / (width * height)
    return (1-val)


# Estimates how much of the space can be reached by computing how many of the tiles can be stood upon
# Value Range 0-1
# maximise
def negativeSpace(x, netG, dim):
    im = translateLatentVector(x, netG, dim)
    unique, counts = numpy.unique(im, return_counts=True)
    dist = dict(zip(unique, counts))
    width = numpy.shape(im)[1]
    height = numpy.shape(im)[0]
    val = 0
    # Tiles you can stand on
    val += dist.get(GROUND, 0)
    val += dist.get(BREAK, 0)
    val += dist.get(QUESTIONC, 0)
    val += dist.get(QUESTIONP, 0)
    val += dist.get(TUBE, 0) * 2 # Because only one tile, but width of 2
    val += dist.get(PLANT, 0) * 2 # Because only one tile, but width of 2
    val += dist.get(BILL, 0)
    val = float(val) / (width * height)
    return (1-val)


# Frequency of pretty tiles, i.e. non-standard.
# Value Range 0-1
# maximise
def decorationFrequency(x, netG, dim):
    im = translateLatentVector(x, netG, dim)
    unique, counts = numpy.unique(im, return_counts=True)
    dist = dict(zip(unique, counts))
    width = numpy.shape(im)[1]
    height = numpy.shape(im)[0]
    val = 0
    # Pretty tiles according to paper
    val += dist.get(BREAK, 0)
    val += dist.get(QUESTIONC, 0)
    val += dist.get(QUESTIONP, 0)
    val += dist.get(COIN, 0)
    val += dist.get(TUBE, 0)
    val += dist.get(PLANT, 0)
    val += dist.get(BILL, 0)
    val += dist.get(GOOMBA, 0)
    val += dist.get(GKOOPA, 0)
    val += dist.get(RKOOPA, 0)
    val += dist.get(SPINY, 0)
    val = float(val) / (width * height)
    return (1-val)

# gets vertical distribution of tiles you can stand on
# Value range ?
# maximise
def positionDistribution(x, netG, dim):
    im = translateLatentVector(x, netG, dim)
    height = numpy.shape(im)[0]
    xm, xs, ym, ys = tilePositionSummaryStats(im, [GROUND, BREAK, QUESTIONP, QUESTIONC, TUBE, PLANT, BILL])
    print(numpy.std(numpy.array([0, height-1])))
    return (-ys)

# get horizontal distribution of enemies
# Value range ?
# maximise
def enemyDistribution(x, netG, dim):
    im = translateLatentVector(x, netG, dim)
    xm, xs, ym, ys = tilePositionSummaryStats(im, [PLANT, BILL, GOOMBA, GKOOPA, RKOOPA, SPINY])
    return (-xs)

def translateLatentVector(x, netG, dim):
    ##Fix for new pytorch compatibility below (from Jacob)
    # This is a new DCGAN model that has the proper state dict labels/keys for the latest version of PyTorch (no periods '.')
    generator = dcgan.DCGAN_G(imageSize, dim, features, ngf, ngpu, n_extra_layers)
    # This is a state dictionary with deprecated key labels/names
    deprecatedModel = torch.load(netG, map_location=lambda storage, loc: storage)
    # Make new model with weights/parameters from deprecatedModel but labels/keys from generator.state_dict()
    fixedModel = OrderedDict()
    for (goodKey,ignore) in list(generator.state_dict().items()):
    # Take the good key and replace the : with . in order to get the deprecated key so the associated value can be retrieved
        badKey = goodKey.replace(":",".")
        # Some parameter settings of the generator.state_dict() are not actually part of the saved models
        if badKey in deprecatedModel:
            goodValue = deprecatedModel[badKey]
            fixedModel[goodKey] = goodValue

    if not fixedModel:
        # If the fixedModel was empty, then the model was trained with the new labels, and the regular load process is fine
        generator.load_state_dict(deprecatedModel)
    else:
        # Load the parameters with the fixed labels  
        generator.load_state_dict(fixedModel)

    inp = numpy.array_split(x, len(x) / dim)
    final = None
    for x in inp:
        latent_vector = torch.FloatTensor(x).view(batchSize, dim, 1, 1)
        with torch.no_grad():
            levels = generator(Variable(latent_vector))
        levels.data = levels.data[:, :, :14, :28]
        im = levels.data.cpu().numpy()
        im = numpy.argmax(im, axis=1)
        if final is None:
            final = im[0]
        else:
            final = numpy.column_stack([final, im[0]])
    return final



def progressSimAStar(x, netG, dim):
    return executeSimulation(x, netG, dim, 0, 0)
def basicFitnessSimAStar(x, netG, dim):
    return executeSimulation(x, netG, dim, 1, 0)
def airTimeSimAStar(x, netG, dim):
    return executeSimulation(x, netG, dim, 2, 0)
def timeTakenSimAStar(x, netG, dim):
    return executeSimulation(x, netG, dim, 3, 0)
def progressSimScared(x, netG, dim):
    return executeSimulation(x, netG, dim, 0, 1)
def basicFitnessSimScared(x, netG, dim):
    return executeSimulation(x, netG, dim, 1, 1)
def airTimeSimScared(x, netG, dim):
    return executeSimulation(x, netG, dim, 2, 1)
def timeTakenSimScared(x, netG, dim):
    return executeSimulation(x, netG, dim, 3, 1)


def decodeProblem(problem):
    available_jsons = ["overworld", "underground"]  # G
    available_fit = [enemyDistribution, positionDistribution, decorationFrequency, negativeSpace, leniency,
                     basicFitnessSimAStar, basicFitnessSimAStar, basicFitnessSimScared,
                     airTimeSimAStar, airTimeSimAStar, airTimeSimScared,
                     timeTakenSimAStar, timeTakenSimAStar, timeTakenSimScared]  # F
    available_c = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]

    f = int(problem / len(available_jsons))
    c = available_c[f]
    g = problem % len(available_jsons)
    return c, available_jsons[g], available_fit[f]
    

def biProbSplitter(problem_id):
    if problem_id==1:
        return [4, 6]
    elif problem_id==2:
        return [4, 8]    
    elif problem_id==3:
        return [11, 17]
    elif problem_id==4:
        return [11, 23]
    elif problem_id==5:
        return [12, 18]
    elif problem_id==6:
        return [12, 24]
    elif problem_id==7:
        return [13, 19]
    elif problem_id==8:
        return [13, 25]
    elif problem_id==9:
        return [14, 20]
    elif problem_id==10:
        return [14, 26]
    else:
        raise ValueError('Suite {} has no function {}'.format("mario-gan-biobj", problem_id))

def getNetG(problem, inst, dim, c, json):
    if c == 1:
        dim = 5

    # print(path)
    pattern = "{}/GAN/{}-{}-{}/netG_epoch_*_{}.pth".format(path,json, dim, budget,
                                                            inst)
    files = glob.glob(pattern)

    epochs = [int(str.split(os.path.basename(file), "_")[2]) for file in files]
    netG = "{}/GAN/{}-{}-{}/netG_epoch_{}_{}.pth".format(path, json, dim, budget, max(epochs),
                                                          inst)
    return netG, dim


def evaluate_mario_gan(suite_name, problem, inst, x):
    available_dims = [10, 20, 30, 40]
    available_instances = [5641, 3854, 8370, 494, 1944, 9249, 2517]
    if len(x) not in available_dims:  # check Dimension available
        raise ValueError("x was dimension '{}', but is not available".format(len(x)))
    if inst < 0 | inst > len(available_instances):
        raise ValueError("asked for instance '{}', but is not available".format(inst))

    probs = [problem]
    if suite_name == 'mario-gan-biobj':
        probs = [j - 1 for j in biProbSplitter(problem)]

    out = [None] * len(probs)
    for i, prob in enumerate(probs):
        c, json, fun = decodeProblem(prob - 1)  # -1 because COCO starts with index 1
        netG, d = getNetG(prob - 1, available_instances[inst - 1], len(x), c, json)  # -1 because COCO starts with index 1
        out[i] = fun(x, netG, d)

    return out


if __name__ == '__main__':
    out = evaluate_mario_gan(
        "mario-gan", 3, 1, [0.577396866201949, 0.7814522617215477, -0.4290037786827649,
                             -0.7939910428259774, 0.4272655228644559, -0.4788319759161429,
                             0.7092257647567968, -0.7713656070501105, 0.751081985876608,
                             -0.7008837870643055])
    print(out)

