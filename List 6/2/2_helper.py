import numpy as np
import math

rules = np.random.random((3, 3))
print rules

with open('/workspaces/pyneurgen/2/satelite_data/ImageExpertReduced.txt', 'r') as expert_file:
    ground_truth = expert_file.read()
ground_truth = np.array([int(float(x)) for x in ground_truth.split()])

with open('/workspaces/pyneurgen/2/satelite_data/ImageRawReduced.txt', 'r') as data_file:
    satelite_images = data_file.read()
satelite_images = np.array([[float(x) for x in line.split()] for line in satelite_images.splitlines()])

signal = np.matmul(rules, satelite_images)

predictions = signal.argmax(axis=0) + 1

fitness = (predictions == ground_truth).sum()

head_gt = ground_truth[:10]
print head_gt.tolist()

head_si = satelite_images[:,:10]
print head_si.tolist()
print fitness