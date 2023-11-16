#!/usr/bin/env python3
import argparse
import itertools

import umbridge
import numpy as np


parser = argparse.ArgumentParser(description='Model output test.')
parser.add_argument(
    'url', metavar='url', type=str,
    help='the ULR on which the model is running, for example {0}'.format(
        'http://localhost:4242'))
args = parser.parse_args()
print(f"Connecting to host URL {args.url}")

model = umbridge.HTTPModel(args.url, "genz")

assert model.get_input_sizes({"nvars": 4}) == [4]
assert model.get_output_sizes() == [1]

# check all combinations of config runs
names = ["oscillatory", "product_peak", "corner_peak", "gaussian",
         "c0continuous", "discontinuous"]
nvars = np.arange(2, 7).astype(float)
decays = ["none", "quadratic", "quartic", "exp", "sqexp"]
test_scenarios = itertools.product(*[names, nvars, decays])
for test_scenario in test_scenarios:
    np.random.seed(1)
    parameters = [np.random.uniform(0, 1, (int(test_scenario[1]))).tolist()]
    config = {"name": test_scenario[0], "nvars": test_scenario[1],
              "coef_type": test_scenario[2]}
    value = model(parameters, config)
