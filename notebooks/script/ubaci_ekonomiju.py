import pandas as pd
import numpy as np

import math
import time



def trend(referentni_datum, zadnji_datum, vrsta_proizvoda, indikator, ekonomski_indikatori):
    maks_godina = 2019 if indikator == 'Cijena nafte' else 2024

    ref_godina = referentni_datum.year
    zadnja_godina = zadnji_datum.year
    
    if ref_godina == zadnja_godina:
        zadnja_godina += 2

    if zadnja_godina > maks_godina:
        zadnja_godina = maks_godina

    serija = ekonomski_indikatori.loc[ref_godina:zadnja_godina, indikator]
    
    y = np.array(serija)
    x = np.array(serija.index)
    
    f = np.polyfit(x, y, 1)
    
    if np.absolute(f[0]) < 1e-10:
        return 0
    
    return f[0] if vrsta_proizvoda == 'A' else -f[0]

def std(referentni_datum, zadnji_datum, indikator, ekonomski_indikatori):
    maks_godina = 2019 if indikator == 'Cijena nafte' else 2024

    ref_godina = referentni_datum.year
    zadnja_godina = zadnji_datum.year
    
    if ref_godina == zadnja_godina:
        zadnja_godina += 2

    if zadnja_godina > maks_godina:
        zadnja_godina = maks_godina

    serija = ekonomski_indikatori.loc[ref_godina:zadnja_godina, indikator]
    
    return np.std(np.array(serija))

def udaljenost_od_prosjeka(referentni_datum, zadnji_datum, vrsta_proizvoda, indikator, ekonomski_indikatori):
    maks_godina = 2019 if indikator == 'Cijena nafte' else 2024

    ref_godina = referentni_datum.year
    zadnja_godina = zadnji_datum.year
    
    if ref_godina == zadnja_godina:
        zadnja_godina += 2
        
    if zadnja_godina > maks_godina:
        zadnja_godina = maks_godina

    serija = ekonomski_indikatori.loc[ref_godina:zadnja_godina, indikator]
    
    np_wrapper = np.array(serija)
    prosjek = np.mean(np_wrapper)
    
    return np_wrapper[0] - prosjek if vrsta_proizvoda == 'A' else prosjek - np_wrapper[0]