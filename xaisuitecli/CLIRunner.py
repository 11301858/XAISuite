#!/usr/bin/env python
from xaisuite import*
import sys

model = sys.argv[1]
data = sys.argv[2]
target = sys.argv[3]

train_and_explainModel(model, load_data_CSV(data, target))