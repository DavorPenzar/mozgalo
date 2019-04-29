# -*- coding: utf-8 -*-

"""
Skripta s funkcijama korisnima za mjerenje vremena.

"""

# Standardna Python biblioteka.
import copy as _copy
import datetime as _datetime
import functools as _functools
import math as _math
import os as _os
import random as _random
import six as _six
import string as _string
import sys as _sys
import time as _time
import warnings as _warnings

# SciPy paketi.
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import scipy as _sp
import sympy as _sym

# Seaborn.
import seaborn as _sns

# Definicija funkcije hms_time.
def hms_time (duration):
    """
    Izrazi vrijeme u sekundama kao vrijeme u satima, sekundama i minutama.

    Argument duration iznos je vremena u sekundama, a povratna vrijednost je
    rjecnik s kljucevima
        --  'h' --  najvece cijelo broja sati (int),
        --  'm' --  najvece cijelo broja minuta u intervalu [0, 60) (int),
        --  's' --  broj sekundi u intervalu [0, 60) (float)
    pri cemu se uzima apsolutna vrijednost argumenta duration (ako je
    duration < 0, potrebno je vrijednosti povratnog rjecnika pomnoziti s -1).

    """

    # Dohvati apsolutnu vrijednost argumenta duration.
    duration = _math.fabs(duration)

    # Vrati trauzeni rjecnik.
    return {
        'h' : int(_math.floor(duration / 3600.0)),
        'm' : int(
            _math.floor(
                (duration - 3600.0 * int(_math.floor(duration / 3600.0))) / 60.0
            )
        ),
        's' : float(duration - 60 * int(_math.floor(duration / 60.0)))
    }
