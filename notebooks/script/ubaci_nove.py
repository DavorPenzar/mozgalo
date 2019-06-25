# -*- coding: utf-8 -*-

import numpy as _np

_pocetak_krize = 2008
_kraj_krize = 2015

_prva_optimisticna_godina = 2016
_zadnja_optimisticna_godina = 2018

def trend (
    vrsta_proizvoda,
    referentni_datum,
    zadnji_datum,
    indikator,
    ekonomski_indikatori
):
    maks_godina = 2019 if indikator == 'Cijena nafte' else 2024

    ref_godina = referentni_datum.year
    zadnja_godina = zadnji_datum.year

    if ref_godina == zadnja_godina:
        zadnja_godina += 2

    zadnja_godina = _np.minimum(zadnja_godina, maks_godina)

    serija = ekonomski_indikatori.loc[ref_godina:zadnja_godina, indikator]

    y = _np.asarray(serija)
    x = _np.asarray(serija.index)

    f = _np.polyfit(x, y, 1)

    if _np.abs(f[0]) < 1.0e-10:
        return 0

    return (
        f[0] if vrsta_proizvoda == 'A'
            else -f[0] if vrsta_proizvoda == 'L'
            else None
    )

def std (
    referentni_datum,
    zadnji_datum,
    indikator,
    ekonomski_indikatori
):
    maks_godina = 2019 if indikator == 'Cijena nafte' else 2024

    ref_godina = referentni_datum.year
    zadnja_godina = zadnji_datum.year

    if ref_godina == zadnja_godina:
        zadnja_godina += 2

    zadnja_godina = _np.minimum(zadnja_godina, maks_godina)

    serija = ekonomski_indikatori.loc[ref_godina:zadnja_godina, indikator]

    return _np.std(_np.asarray(serija))

def udaljenost_od_prosjeka (
    vrsta_proizvoda,
    referentni_datum,
    zadnji_datum,
    indikator,
    ekonomski_indikatori
):
    maks_godina = 2019 if indikator == 'Cijena nafte' else 2024

    ref_godina = referentni_datum.year
    zadnja_godina = zadnji_datum.year

    if ref_godina == zadnja_godina:
        zadnja_godina += 2

    zadnja_godina = _np.minimum(zadnja_godina, maks_godina)

    serija = ekonomski_indikatori.loc[ref_godina:zadnja_godina, indikator]

    np_wrapper = _np.asarray(serija)
    prosjek = _np.mean(np_wrapper)

    return (
        np_wrapper[0] - prosjek if vrsta_proizvoda == 'A'
            else prosjek - np_wrapper[0] if vrsta_proizvoda == 'L'
            else None
    )

def ekonomija (df, ekonomski_indikatori):
    df['TREND_bdp'] = df.apply(
        lambda x : trend(
            x.VRSTA_PROIZVODA,
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'BDP (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['TREND_inflacija'] = df.apply(
        lambda x :trend(
            x.VRSTA_PROIZVODA,
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Inflacija (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['TREND_nezaposlenosti'] = df.apply(
        lambda x : trend(
            x.VRSTA_PROIZVODA,
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Stopa nezaposlenosti (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['TREND_nafta'] = df.apply(
        lambda x : trend(
            x.VRSTA_PROIZVODA,
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Cijena nafte',
            ekonomski_indikatori
        ),
        axis = 1
    )

    df['STD_bdp'] = df.apply(
        lambda x : std(
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'BDP (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['STD_inflacija'] = df.apply(
        lambda x : std(
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Inflacija (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['STD_nezaposlenosti'] = df.apply(
        lambda x : std(
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Stopa nezaposlenosti (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['STD_nafta'] = df.apply(
        lambda x : std(
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Cijena nafte',
            ekonomski_indikatori
        ),
        axis = 1
    )

    df['PROSJEK_bdp'] = df.apply(
        lambda x : udaljenost_od_prosjeka(
            x.VRSTA_PROIZVODA,
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'BDP (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['PROSJEK_inflacija'] = df.apply(
        lambda x : udaljenost_od_prosjeka(
            x.VRSTA_PROIZVODA,
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Inflacija (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['PROSJEK_nezaposlenosti'] = df.apply(
        lambda x : udaljenost_od_prosjeka(
            x.VRSTA_PROIZVODA,
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Stopa nezaposlenosti (%)',
            ekonomski_indikatori
        ),
        axis = 1
    )
    df['PROSJEK_nafta'] = df.apply(
        lambda x : udaljenost_od_prosjeka(
            x.VRSTA_PROIZVODA,
            x.ZADNJI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA,
            'Cijena nafte',
            ekonomski_indikatori
        ),
        axis = 1
    )

    return df

def godine_u_krizi (prvi_datum, zadnji_datum):
    prva_godina = prvi_datum.year
    zadnja_godina = zadnji_datum.year

    if prva_godina < _pocetak_krize and zadnja_godina < _pocetak_krize:
        return 0
    if (
        prva_godina < _pocetak_krize and
        zadnja_godina >= _pocetak_krize and
        zadnja_godina <= _kraj_krize
    ):
        return zadnja_godina - _pocetak_krize + 1
    if prva_godina < _pocetak_krize and zadnja_godina > _kraj_krize:
        return _kraj_krize - _pocetak_krize + 1
    if (
        prva_godina >= _pocetak_krize and
        prva_godina <= _kraj_krize and
        zadnja_godina <= _kraj_krize
    ):
        return zadnja_godina - prva_godina + 1
    if (
        prva_godina >= _pocetak_krize and
        prva_godina <= _kraj_krize and
        zadnja_godina > _kraj_krize
    ):
        return _kraj_krize - prva_godina + 1

    return 0

def godine_nakon_krize (planirani_datum_zatvaranja):
    zadnja_godina = planirani_datum_zatvaranja.year - 1

    return int(
        zadnja_godina >= _prva_optimisticna_godina and
        zadnja_godina <= _zadnja_optimisticna_godina
    )

def udaljenost_visine_kamate (vrsta_proizvoda, visina_kamate):
    if vrsta_proizvoda == 'A':
        if visina_kamate < 1.4:
            return _np.maximum(_np.abs(visina_kamate - 0.7), 1.0e-6)
        if visina_kamate >= 11.9:
            return _np.maximum(_np.abs(visina_kamate - 12.6), 1.0e-6)
        else:
            return _np.maximum(_np.abs(visina_kamate - 6.55), 1.0e-6)
    elif vrsta_proizvoda == 'L':
        if visina_kamate >= 6.488:
            return _np.maximum(_np.abs(visina_kamate - 6.89), 1.0e-6)
        else:
            return _np.maximum(_np.abs(visina_kamate - 3.58), 1.0e-6)

    return _np.inf

def mean_probit_kam (
    vrsta_proizvoda,
    vrsta_klijenta,
    proizvod,
    visina_kamate,
    score_vk,
    score_p
):
    skalar = udaljenost_visine_kamate(vrsta_proizvoda, visina_kamate)

    if vrsta_proizvoda == 'A':
        pass
    elif vrsta_proizvoda == 'L':
        skalar **= -1
    else:
        skalar = 1

    index = 'probit(score)-{0:s}'.format(vrsta_proizvoda)

    return _np.log(
        _np.exp(
            (
                score_vk.loc[index, str(vrsta_klijenta)] +
                score_p.loc[index, str(proizvod)]
            ) / 2
        ) / skalar
    )

def mean_score_kam (
    vrsta_proizvoda,
    vrsta_klijenta,
    proizvod,
    visina_kamate,
    score_vk,
    score_p
):
    skalar = udaljenost_visine_kamate(vrsta_proizvoda, visina_kamate)

    if vrsta_proizvoda == 'A':
        pass
    elif vrsta_proizvoda == 'L':
        skalar **= -1
    else:
        skalar = 1

    index = 'score-{0:s}'.format(vrsta_proizvoda)

    return _np.log(
        _np.exp(
            (
                score_vk.loc[index, str(vrsta_klijenta)] +
                score_p.loc[index, str(proizvod)]
            ) / 2
        ) / skalar
    )

def mean_probit_skal (
    vrsta_proizvoda,
    vrsta_klijenta,
    proizvod,
    skalar,
    score_vk,
    score_p
):
    index = 'probit(score)-{0:s}'.format(vrsta_proizvoda)

    return (
        skalar *
        (
            score_vk.loc[index, str(vrsta_klijenta)] +
            score_p.loc[index, str(proizvod)]
        ) / 2
    )

def mean_score_skal (
    vrsta_proizvoda,
    vrsta_klijenta,
    proizvod,
    skalar,
    score_vk,
    score_p
):
    index = 'score-{0:s}'.format(vrsta_proizvoda)

    return (
        skalar *
        (
            score_vk.loc[index, str(vrsta_klijenta)] +
            score_p.loc[index, str(proizvod)]
        ) / 2
    )

def dodavanje (df, ekonomski_indikatori, score_vk, score_p):
    df['duljina_dani'] = (
        df.ZADNJI_PLANIRANI_DATUM_ZATVARANJA - df.ZADNJI_DATUM_OTVARANJA
    ).dt.days
    df['duljina_godine'] = df.duljina_dani / 365.2475

    df['slozeni_kamatni'] = (
        (1 + df.PRVA_VISINA_KAMATE) ** (df.duljina_godine) *
        df.PRVI_UGOVORENI_IZNOS
    )
    df['slozeni_kamatni_po_danu'] = df.slozeni_kamatni / df.duljina_dani
    df['slozeni_kamatni_po_godini'] = df.slozeni_kamatni / df.duljina_godine

    df['jednostavni_kamatni'] = (
        df.ZADNJI_UGOVORENI_IZNOS +
        (
            df.ZADNJI_UGOVORENI_IZNOS *
            df.ZADNJA_VISINA_KAMATE *
            df.duljina_godine
        ) / 100
    )
    df['jednostavni_kamatni_po_danu'] = df.jednostavni_kamatni / df.duljina_dani
    df['jednostavni_kamatni_po_godini'] = (
        df.jednostavni_kamatni / df.duljina_godine
    )

    df['log_zadnji_ugovoreni_iznos'] = _np.log(df.ZADNJI_UGOVORENI_IZNOS)
    df['visina_kamate_kvadrat'] = (
        df.ZADNJA_VISINA_KAMATE * _np.abs(df.ZADNJA_VISINA_KAMATE)
    )

    df['broj_godina_u_krizi'] = df.apply(
        lambda x : godine_u_krizi(
            x.PRVI_DATUM_OTVARANJA,
            x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA
        ),
        axis = 1
    )
    df['krizne_puta_ukupno'] = df.broj_godina_u_krizi * df.duljina_godine
    df['veci_krizni_dug'] = (
        1.5 * df.broj_godina_u_krizi * df.jednostavni_kamatni_po_godini
    )
    df['optimisticne_godine'] = df.apply(
        lambda x : godine_nakon_krize(x.ZADNJI_PLANIRANI_DATUM_ZATVARANJA),
        axis = 1
    )

    df['mean_probit_kam'] = df.apply(
        lambda x : mean_probit_kam(
            x.VRSTA_PROIZVODA,
            x.VRSTA_KLIJENTA,
            x.PROIZVOD,
            x.ZADNJA_VISINA_KAMATE,
            score_vk,
            score_p
        ),
        axis = 1
    )
    df['mean_score_kam'] = df.apply(
        lambda x : mean_score_kam(
            x.VRSTA_PROIZVODA,
            x.VRSTA_KLIJENTA,
            x.PROIZVOD,
            x.ZADNJA_VISINA_KAMATE,
            score_vk,
            score_p
        ),
        axis = 1
    )

    df['promjena_ugovorenog_iznosa'] = (
        df.ZADNJI_UGOVORENI_IZNOS == df.PRVI_UGOVORENI_IZNOS
    ).astype(int)
    df['promjena_tipa_kamate'] = (
        df.ZADNJI_TIP_KAMATE == df.PRVI_TIP_KAMATE
    ).astype(int)
    df['promjena_visine_kamate'] = (
        df.ZADNJA_VISINA_KAMATE == df.PRVA_VISINA_KAMATE
    ).astype(int)

    return df
