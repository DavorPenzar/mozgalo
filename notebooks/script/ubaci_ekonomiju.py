import pandas as pd
import numpy as np

import math
import time

def trend(referentni_datum, zadnji_datum, vrsta_proizvoda, indikator):
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

def std(referentni_datum, zadnji_datum, indikator):
    maks_godina = 2019 if indikator == 'Cijena nafte' else 2024

    ref_godina = referentni_datum.year
    zadnja_godina = zadnji_datum.year
    
    if ref_godina == zadnja_godina:
        zadnja_godina += 2

    if zadnja_godina > maks_godina:
        zadnja_godina = maks_godina

    serija = ekonomski_indikatori.loc[ref_godina:zadnja_godina, indikator]
    
    return np.std(np.array(serija))

def udaljenost_od_prosjeka(referentni_datum, zadnji_datum, vrsta_proizvoda, indikator):
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




t0 = time.time()

ekonomski_indikatori = pd.read_csv('../../ekonomski-pokazatelji/ekonomski_indikatori.csv', index_col=0)
df = pd.read_pickle('../../data/eval/spljosteni.pkl')

print('Ubacivanje zapocinje!\n')
## trendovi
df['TREND_bdp'] = df.apply(lambda x: trend(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], x['VRSTA_PROIZVODA'], 'BDP (%)'), axis=1)
df['TREND_inflacija'] = df.apply(lambda x: trend(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], x['VRSTA_PROIZVODA'], 'Inflacija (%)'), axis=1)
df['TREND_nezaposlenosti'] = df.apply(lambda x: trend(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], x['VRSTA_PROIZVODA'], 'Stopa nezaposlenosti (%)'), axis=1)
df['TREND_nafta'] = df.apply(lambda x: trend(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], x['VRSTA_PROIZVODA'], 'Cijena nafte'), axis=1)
print('Zavrseno "ubacivanje" trend-ova')

## std-ovi
df['STD_bdp'] = df.apply(lambda x: std(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], 'BDP (%)'), axis=1)
df['STD_inflacija'] = df.apply(lambda x: std(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], 'Inflacija (%)'), axis=1)
df['STD_nezaposlenosti'] = df.apply(lambda x: std(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], 'Stopa nezaposlenosti (%)'), axis=1)
df['STD_nafta'] = df.apply(lambda x: std(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], 'Cijena nafte'), axis=1)
print('Zavrseno "ubacivanje" std-ova')

## udaljenost prosjeka od referentne vrijednosti
df['PROSJEK_bdp'] = df.apply(lambda x: udaljenost_od_prosjeka(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], x['VRSTA_PROIZVODA'], 'BDP (%)'), axis=1)
df['PROSJEK_inflacija'] = df.apply(lambda x: udaljenost_od_prosjeka(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], x['VRSTA_PROIZVODA'], 'Inflacija (%)'), axis=1)
df['PROSJEK_nezaposlenosti'] = df.apply(lambda x: udaljenost_od_prosjeka(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], x['VRSTA_PROIZVODA'], 'Stopa nezaposlenosti (%)'), axis=1)
df['PROSJEK_nafta'] = df.apply(lambda x: udaljenost_od_prosjeka(x['ZADNJI_DATUM_OTVARANJA'], x['ZADNJI_PLANIRANI_DATUM_ZATVARANJA'], x['VRSTA_PROIZVODA'], 'Cijena nafte'), axis=1)
print('Zavrseno "ubacivanje" udaljenosti')



df.to_pickle('../../data/eval/spljosteni_sa_ekonomijom.pkl')



# Zavrsi mjerenje vremena.
t1 = time.time()

# Izracunaj vremenski period od t0 do t1 u sekundama.
d = float(t1 - t0)

# Ispisi vrijeme.
print(
    'Trajanje ucitavanja: {h:d}h {m:02d}m {s:06.3f}s ({S:.3f}s)'.format(
        h = int(math.floor(d / 3600)),
        m = int(math.floor((d - 3600 * int(math.floor(d / 3600))) / 60)),
        s = d - 60 * int(math.floor(d / 60)),
        S = d
    )
)
