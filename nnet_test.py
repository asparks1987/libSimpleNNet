from nnet import silvernnet
import numpy as np
from pairgenerator import PairGenerator

gen = PairGenerator(5,10000)
gen.generate_pairs()
gen.save_to_file('nnettest.pg')
