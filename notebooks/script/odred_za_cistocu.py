#!/usr/bin/env python

# -*- coding: utf-8 -*-

import time

import numpy as np
import pandas as pd

from cleaner2 import *

df = pd.read_pickle('../../data/training_dataset_enc.pkl')

t0 = time.time()

df = pocisti_drugacije(df, 3)

t1 = time.time()

print(float(t1 - t0))

df.to_pickle('../../data/spljosteni.pkl')
