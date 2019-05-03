import pandas as pd

glavni_pokazatelji = pd.read_pickle('../../ekonomski-pokazatelji/ekonomski_indikatori.pkl')
place = pd.read_excel('../../ekonomski-pokazatelji/place_i_indeksi.xlsx')


############################################################
################# GLAVNI POKAZATELJI
############################################################


def bdp_hrk(god):
    return glavni_pokazatelji.loc[0, god]

def bdp_eur(god):
    return glavni_pokazatelji.loc[1, god]

def bdp_stanovnik(god):
    return glavni_pokazatelji.loc[2, god]

def bdp_god_stopa(god):
    return glavni_pokazatelji.loc[3, god]

def inflacija(god):
    return glavni_pokazatelji.loc[4, god]

def tekuci_racun_eur(god):
    return glavni_pokazatelji.loc[5, god]

def tekuci_racun_bdp(god):
    return glavni_pokazatelji.loc[6, god]

def izvoz(god):
    return glavni_pokazatelji.loc[7, god]

def uvoz(god):
    return glavni_pokazatelji.loc[8, god]

def inozemni_dug_eur(god):
    return glavni_pokazatelji.loc[9, god]

def inozemni_dug_bdp(god):
    return glavni_pokazatelji.loc[10, god]

def inozemni_dug_izvoz(god):
    return glavni_pokazatelji.loc[11, god]

def otplaceni_inozemni_dug(god):
    return glavni_pokazatelji.loc[12, god]

def med_pricuva_eur(god):
    return glavni_pokazatelji.loc[13, god]

def med_pricuva_uvoz(god):
    return glavni_pokazatelji.loc[14, god]

def devizni_tecaj_eur(god):
    return glavni_pokazatelji.loc[15, god]

def devizni_tecaj_usd(god):
    return glavni_pokazatelji.loc[16, god]

def prosjecni_devizni_tecaj_eur(god):
    return glavni_pokazatelji.loc[17, god]

def prosjecni_devizni_tecaj_usd(god):
    return glavni_pokazatelji.loc[18, god]

def neto_poz_zad_hrk(god):
    return glavni_pokazatelji.loc[19, god]

def neto_poz_zad_bdp(god):
    return glavni_pokazatelji.loc[20, god]

def dug_bdp(god):
    return glavni_pokazatelji.loc[21, god]

def stopa_nezaposlenosti(god):
    return glavni_pokazatelji.loc[22, god]

def stopa_zaposlenosti(god):
    return glavni_pokazatelji.loc[23, god]

############################################################
################# PLACE I INDEKSI
############################################################


mapiranje_mjeseca = {
 '1': 'I',
 '2': 'II',
 '3': 'III',
 '4': 'IV',
 '5': 'V',
 '6': 'VI',
 '7': 'VII',
 '8': 'VIII',
 '9': 'IX',
 '10': 'X',
 '11': 'XI',
 '12': 'XII',
}


def prosjecna_bruto(mjesec, godina):
    mjesec = mapiranje_mjeseca[str(mjesec)]
    godina = str(godina)
    return place.loc[0, mjesec + '. ' + godina + '.']

def prosjecna_neto(mjesec, godina):
    mjesec = mapiranje_mjeseca[str(mjesec)]
    godina = str(godina)
    return place.loc[1, mjesec + '. ' + godina + '.']

def indeks_potr_cijena_gg(mjesec, godina):
    mjesec = mapiranje_mjeseca[str(mjesec)]
    godina = str(godina)
    return place.loc[2, mjesec + '. ' + godina + '.']

def indeks_potr_cijena_mm(mjesec, godina):
    mjesec = mapiranje_mjeseca[str(mjesec)]
    godina = str(godina)
    return place.loc[3, mjesec + '. ' + godina + '.']
