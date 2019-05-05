# -*- coding: utf-8 -*-

"""
Skripta s klasom za racunanje tocnosti prediktivnog modela (binarnog klasifikatora).

"""

# Standardna Python biblioteka.
import copy as _copy
import math as _math
import numbers as _numbers

# Definicija klase Score.
class Score (object):
    """
    Ocjenjivac tocnosti prediktivnog modela (binarnog klasifikatora).

    Arguments
    ---------
    n_positives : int
        Ukupni broj pozitivnih instanci u skupu svih instanci.

    n_negatives : int
        Ukupni broj negativnih instanci u skupu svih instanci.

    """

    def __new__ (cls, n_positives = None, n_negatives = None):
        assert issubclass(cls, Score)

        instance = super(Score, cls).__new__(cls)

        instance._n_positives = 0
        instance._n_negatives = 0

        instance._true_positives = 0
        instance._false_positives = 0
        instance._true_negatives = 0
        instance._false_negatives = 0

        instance._accuracy = 0.0
        instance._precision = 0.0
        instance._recall = 0.0
        instance._specificity = 0.0

        return instance

    def __init__ (self, n_positives, n_negatives):
        assert isinstance(self, Score)

        assert (
            isinstance(n_positives, _numbers.Integral) and
            isinstance(n_negatives, _numbers.Integral)
        )
        assert n_positives > 0 and n_negatives > 0

        instance._n_positives = _copy.deepcopy(int(n_positives))
        instance._n_negatives = _copy.deepcopy(int(n_negatives))

        instance._true_positives = 0
        instance._false_positives = instance._n_positives
        instance._true_negatives = 0
        instance._false_negatives = instance._n_negatives

    def reset (self):
        """
        Ponisti spremljenu eventualnu prethodnu ocjenu.

        Arguments
        ---------
        true_positives : int
            Broj tocno predvidenih pozitivnih instanci.

        true_negatives : int
            Broj tocno predvidenih negativnih instanci.

        Returns
        -------
        self : Score
            Povratna vrijednost je self.

        """

        assert isinstance(self, Score)

        # Ponisti ocjenu.

        self._true_positives = 0
        self._false_positives = self._n_positives
        self._true_negatives = 0
        self._false_negatives = self._n_negatives

        self._accuracy = 0.0
        self._precision = 0.0
        self._recall = 0.0
        self._specificity = 0.0

        return self

    def score (
        self,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives
    ):
        """
        Ocijeni prediktivni model.

        Prije ocjenjivanja poziva se metoda Score.reset stoga je nemoguce u
        istom objektu spremiti ocjene dva ili vise prediktivna modela (za svaki
        model potrebno je kreirati zasebni objekt klase Score).

        Arguments
        ---------
        true_positives : int
            Broj tocno predvidenih pozitivnih instanci.

        false_positives : int
            Broj krivo predvidenih pozitivnih instanci.

        true_negatives : int
            Broj tocno predvidenih negativnih instanci.

        false_negatives : int
            Broj krivo predvidenih negativnih instanci.

        Returns
        -------
        self : Score
            Povratna vrijednost je self.

        """

        assert isinstance(self, Score)

        # Provjeri sve argumente.

        assert (
            isinstance(true_positives, _numbers.Integral) and
            isinstance(false_positives, _numbers.Integral) and
            isinstance(true_negatives, _numbers.Integral) and
            isinstance(false, _numbers.Integral) and
        )
        assert (
            true_positives >= 0 and
            false_positives >= 0 and
            true_negatives >= 0 and
            false_negatives >= 0
        )
        assert (
            true_positives + false_negatives == n_positives and
            false_positives + true_negatives == n_negatives
        )

        # Ponisti ocjenu.
        self.reset()

        # Dohvati vrijednosti.
        self._true_positives = _copy.deepcopy(int(true_positives))
        self._false_positives = _copy.deepcopy(int(false_positives))
        self._true_negatives = _copy.deepcopy(int(true_negatives))
        self._false_negatives = _copy.deepcopy(int(false_positives))

        # Izracunaj novu ocjenu.

        self._accuracy = (
            float(self._true_positives + self._true_negatives) /
            (self._n_positives + self._n_negatives)
        )
        self._precision = (
            float(self._true_positives) /
            (self._true_positives + self._false_positives)
        )
        self._recall = float(self._true_positives) / self._n_positives
        self._specificity = float(self._true_negatives) / self._n_negatives

        return self

    def __repr__ (self):
        assert isinstance(self, Score)

        return (
            '<'
                '{name:s}: ('
                    'P: {P:d}, N: {N:d}; '
                    'TP: {TP:d}, FP: {FP:d}, TN: {TN:d}, FN: {FN:d}'
                ')'
            '>'.format(
                name = self.__class__.__name__,
                P = self._n_positives,
                N = self._n_negatives,
                TP = self._true_positives,
                FP = self._false_positives,
                TN = self._true_negatives,
                FN = self._false_negatives
            )
        )

    def __copy__ (self):
        assert isinstance(self, Score)

        instance = Score(self._n_positives, self._n_negatives)

        instance._true_positives = self._true_positives
        instance._false_positives = self._false_positives
        instance._true_negatives = self._true_negatives
        instance._false_negatives = self._false_negatives

        instance._accuracy = self._accuracy
        instance._precision = self._precision
        instance._recall = self._recall
        instance._specificity = self._specificity

        return instance

    def __deepcopy__ (self, memo = dict()):
        assert isinstance(self, Score)

        instance = Score(
            _copy.deepcopy(self._n_positives, memo),
            _copy.deepcopy(self._n_negatives, memo)
        )

        instance._true_positives = _copy.deepcopy(self._true_positives, memo)
        instance._false_positives = _copy.deepcopy(self._false_positives, memo)
        instance._true_negatives = _copy.deepcopy(self._true_negatives, memo)
        instance._false_negatives = _copy.deepcopy(self._false_negatives, memo)

        instance._accuracy = _copy.deepcopy(self._accuracy, memo)
        instance._precision = _copy.deepcopy(self._precision, memo)
        instance._recall = _copy.deepcopy(self._recall, memo)
        instance._specificity = _copy.deepcopy(self._specificity, memo)

        return instance

    @property
    def n_positives (self):
        """
        Ukupni broj pozitivnih instanci.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._n_positives)

    @property
    def n_negatives (self):
        """
        Ukupni broj negativnih instanci.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._n_negatives)

    @property
    def true_positives (self):
        """
        Broj tocno predvidenih pozitivnih instanci.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._true_positives)

    @property
    def false_positives (self):
        """
        Broj krivo predvidenih pozitivnih instanci.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._false_positives)

    @property
    def true_negatives (self):
        """
        Broj tocno predvidenih negativnih instanci.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._true_negatives)

    @property
    def false_negatives (self):
        """
        Broj krivo predvidenih negativnih instanci.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._false_negatives)

    @property
    def accuracy (self):
        """
        Tocnost prediktivnog modela.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._accuracy)

    @property
    def precision (self):
        """
        Preciznost prediktivnog modela.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._precision)

    @property
    def recall (self):
        """
        Odziv prediktivnog modela.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._recall)

    @property
    def specificity (self):
        """
        Specificnost prediktivnog modela.

        """

        assert isinstance(self, Score)

        return _copy.deepcopy(self._specificity)

    def F (self, beta = 1.0):
        """
        Izracunaj F_beta vrijednost ocjene.

        Arguments
        ---------
        beta : float, optional
            Parametar beta za F_beta ocjenu.  Zadana vrijednost je 1.0.

        Returns
        -------
        f : float
            F_beta vrijednost spremljene ocjene prediktivnog modela.

        """

        assert isinstance(self, Score)

        # Provjeri argument.

        assert isinstance(beta, _numbers.Real)
        assert not (_math.isnan(beta) or _math.isinf(beta)) and beta > 0

        # Izracunaj beta^2.
        beta2 = _copy.deepcopy(float(beta)) ** 2

        del beta

        # Izracunaj i vrati F_beta vrijednost ocjene.
        return (
            (1.0 + beta2) *
            self._precision *
            self._recall /
            (beta2 * self._precision + self._recall)
        )
