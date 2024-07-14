import itertools as it
import numpy as np
import random
from math import log as _log, isnan as _isnan

# Define some useful constants
NEGINF = float("-inf")
INF = float("inf")
eps = np.finfo(np.float64).eps

class DiscreteDistribution:
    """
    A discrete distribution, made up of characters and their probabilities,
    assuming that these probabilities will sum to 1.0.
    """

    @property
    def parameters(self):
        return [self.dist]

    @parameters.setter
    def parameters(self, parameters):
        d = parameters[0]
        self.dist = d
        self.log_dist = {key: _log(value) for key, value in d.items()}

    def __init__(self, characters=None, frozen=False):
        """
        Make a new discrete distribution with a dictionary of discrete
        characters and their probabilities, checking to see that these
        sum to 1.0. Each discrete character can be modelled as a
        Bernoulli distribution.
        """
        if characters is None:
            characters = {}

        self.name = "DiscreteDistribution"
        self.frozen = frozen
        self.is_blank_ = True
        self.dtype = None

        if len(characters) > 0:
            self.is_blank_ = False
            self.dtype = self._get_dtype(characters)
            self.dist = characters.copy()
            self.log_dist = {key: _log(value) for key, value in characters.items()}
            self.summaries = [{key: 0 for key in characters.keys()}, 0]

    def _get_dtype(self, characters):
        """
        Determine dtype from characters.
        """
        return str(type(list(characters.keys())[0])).split()[-1].strip('>').strip("'")

    def __reduce__(self):
        """Serialize the distribution for pickle."""
        return self.__class__, (self.dist, self.frozen)

    def __len__(self):
        return len(self.dist)

    def __mul__(self, other):
        """Multiply this by another distribution sharing the same keys."""
        self_keys = self.keys()
        other_keys = other.keys()
        distribution, total = {}, 0.0

        if isinstance(other, DiscreteDistribution) and self_keys == other_keys:
            self_values = self.dist.values()
            other_values = other.dist.values()

            for key, x, y in zip(self_keys, self_values, other_values):
                if _check_nan(key):
                    distribution[key] = (1 + eps) * (1 + eps)
                else:
                    distribution[key] = (x + eps) * (y + eps)
                total += distribution[key]
        else:
            assert set(self_keys) == set(other_keys)
            self_items = self.dist.items()

            for key, x in self_items:
                if _check_nan(key):
                    x = 1.
                y = other.probability(key)
                distribution[key] = (x + eps) * (y + eps)
                total += distribution[key]

        for key in self_keys:
            distribution[key] /= total
            if distribution[key] <= eps / total:
                distribution[key] = 0.0
            elif distribution[key] >= 1 - eps / total:
                distribution[key] = 1.0

        return DiscreteDistribution(distribution)

    def equals(self, other):
        """Return if the keys and values are equal"""
        if not isinstance(other, DiscreteDistribution):
            return False

        self_keys = self.keys()
        other_keys = other.keys()

        if self_keys == other_keys:
            self_values = self.log_dist.values()
            other_values = other.log_dist.values()

            for key, self_prob, other_prob in zip(self_keys, self_values, other_values):
                if _check_nan(key):
                    continue
                self_prob = round(self_prob, 12)
                other_prob = round(other_prob, 12)
                if self_prob != other_prob:
                    return False
        elif set(self_keys) == set(other_keys):
            self_items = self.log_dist.items()

            for key, self_prob in self_items:
                if _check_nan(key):
                    self_prob = 0.
                else:
                    self_prob = round(self_prob, 12)
                other_prob = round(other.log_probability(key), 12)
                if self_prob != other_prob:
                    return False
        else:
            return False

        return True

    def clamp(self, key):
        """Return a distribution clamped to a particular value."""
        return DiscreteDistribution({k: 0. if k != key else 1. for k in self.keys()})

    def keys(self):
        """Return the keys of the underlying dictionary."""
        return tuple(self.dist.keys())

    def items(self):
        """Return items of the underlying dictionary."""
        return tuple(self.dist.items())

    def values(self):
        """Return values of the underlying dictionary."""
        return tuple(self.dist.values())

    def mle(self):
        """Return the maximally likely key."""
        max_key, max_value = None, 0
        for key, value in self.items():
            if value > max_value:
                max_key, max_value = key, value
        return max_key

    def bake(self, keys):
        """Encoding the distribution into integers."""
        if keys is None:
            return

        n = len(keys)
        self.encoded_keys = keys
        self.encoded_counts = np.zeros(n, dtype='float64')
        self.encoded_log_probability = np.full(n, NEGINF, dtype='float64')

        for i in range(n):
            key = keys[i]
            self.encoded_log_probability[i] = self.log_dist.get(key, NEGINF)

    def probability(self, X):
        """Return the prob of the X under this distribution."""
        return self.__probability(X)

    def __probability(self, X):
        if _check_nan(X):
            return 1.
        else:
            return self.dist.get(X, 0)

    def log_probability(self, X):
        """Return the log prob of the X under this distribution."""
        return self.__log_probability(X)

    def __log_probability(self, X):
        if _check_nan(X):
            return 0.
        else:
            return self.log_dist.get(X, NEGINF)

    def _log_probability(self, X, log_probability, n):
        for i in range(n):
            if _isnan(X[i]):
                log_probability[i] = 0.
            elif X[i] < 0 or X[i] > self.n:
                log_probability[i] = NEGINF
            else:
                log_probability[i] = self.encoded_log_probability[X[i]]

    def sample(self, n=None, random_state=None):
        random_state = check_random_state(random_state)
        keys = list(self.dist.keys())
        probabilities = list(self.dist.values())

        if n is None:
            return random_state.choice(keys, p=probabilities)
        else:
            return random_state.choice(keys, p=probabilities, size=n)

    def fit(self, items, weights=None, inertia=0.0, pseudocount=0.0):
        """
        Set the parameters of this Distribution to maximize the likelihood of
        the given sample. Items holds some sort of sequence. If weights is
        specified, it holds a sequence of value to weight each item by.
        """
        if self.frozen:
            return self

        self.summarize(items, weights)
        self.from_summaries(inertia, pseudocount)
        return self

    def summarize(self, items, weights=None):
        """Reduce a set of observations to sufficient statistics."""
        if weights is None:
            weights = np.ones(len(items))
        else:
            weights = np.asarray(weights)

        items = np.asarray(items).flatten()

        for i in range(len(items)):
            x = items[i]
            if not _check_nan(x):
                try:
                    self.summaries[0][x] += weights[i]
                except KeyError:
                    self.summaries[0][x] = weights[i]
                self.summaries[1] += weights[i]

    def from_summaries(self, inertia=0.0, pseudocount=0.0):
        """Use the summaries in order to update the distribution."""
        if self.summaries[1] == 0 or self.frozen:
            return

        values = self.summaries[0].values()
        _sum = sum(values) + pseudocount * len(values)

        for key, value in self.summaries[0].items():
            value += pseudocount
            try:
                self.dist[key] = self.dist[key] * inertia + (1 - inertia) * (value / _sum)
            except KeyError:
                self.dist[key] = value / _sum
            self.log_dist[key] = _log(self.dist[key])

        self.bake(self.encoded_keys)

        if self.is_blank_:
            self.dtype = self._get_dtype(self.dist)
            self.is_blank_ = False

        self.clear_summaries()

    def clear_summaries(self):
        """Clear the summary statistics stored in the object."""
        self.summaries = [{key: 0 for key in self.keys()}, 0]
        if hasattr(self, 'encoded_counts'):
            self.encoded_counts.fill(0)

    def to_dict(self):
        return {
            'class': 'Distribution',
            'dtype': self.dtype,
            'name': self.name,
            'parameters': [{str(key): value for key, value in self.dist.items()}],
            'frozen': self.frozen
        }

    @classmethod
    def from_samples(cls, items, weights=None, pseudocount=0, keys=None):
        """Fit a distribution to some data without pre-specifying it."""
        key_initials = {}
        if keys is not None:
            clean_keys = tuple(key for key in keys if not _check_nan(key))
            # A priori equal probability.
            key_initials = {key: 1. / len(clean_keys) for key in clean_keys}
        return cls(key_initials).fit(items, weights=weights, pseudocount=pseudocount)

    @classmethod
    def blank(cls):
        return cls()

def _check_nan(X):
    """Checks to see if a value is nan, either as a float or a string."""
    if isinstance(X, (str, np.str_)):
        return X == 'nan'
    if isinstance(X, (float, np.float32, np.float64)):
        return np.isnan(X)
    return X is None

def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance.

    This function will check to see whether the input seed is a valid seed
    for generating random numbers. This is a slightly modified version of
    the code from sklearn.utils.validation.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)