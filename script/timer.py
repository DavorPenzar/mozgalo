# -*- coding: utf-8 -*-

"""
Skripta s funkcijama korisnima za mjerenje vremena.

"""

# Standardna Python biblioteka
import math as _math
import time as _time

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
