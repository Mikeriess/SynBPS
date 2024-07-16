import numpy as np
import numbers
import itertools as it

from SynBPS.simulation.Memory_process.distributions.JointProbabilityTable import JointProbabilityTable
from SynBPS.simulation.Memory_process.distributions.Distribution import MultivariateDistribution

import math

def _log(x):
    """
    A wrapper for the math.log function, returning negative infinity if the input is 0.
    """
    return math.log(x) if x > 0 else float('-inf')


def _check_nan(X):
    """
    Checks to see if a value is NaN, either as a float or a string.
    """
    if isinstance(X, (str, np.str_)):
        return X == 'nan'
    if isinstance(X, (float, np.float32, np.float64)):
        return math.isnan(X)
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
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)


class ConditionalProbabilityTable(MultivariateDistribution):
    """A conditional probability table, which is dependent on values from at least one previous distribution but up to as many as you want to encode for."""

    def __init__(self, table, parents=None, frozen=False):
        """Take in the distribution represented as a list of lists, where each inner list represents a row."""
        self.name = "ConditionalProbabilityTable"
        self.m = len(parents) if parents is not None else len(table[0]) - 2
        self.n = len(table)
        self.k = len(set(row[-2] for row in table))
        self.idxs = [0] * (self.m + 1)
        self.marginal_idxs = [0] * self.m
        self.values = np.zeros(self.n, dtype='float64')
        self.counts = np.zeros(self.n, dtype='float64')
        self.marginal_counts = np.zeros(self.n // self.k, dtype='float64')
        self.column_idxs = np.arange(self.m + 1, dtype='int32')
        self.n_columns = self.m + 1
        self.dtypes = [str(type(column)).split()[-1].strip('>').strip("'") for column in table[0]]

        self.idxs[0] = 1
        self.idxs[1] = self.k
        for i in range(self.m - 1):
            k = len(np.unique([row[self.m - i - 1] for row in table]))
            self.idxs[i + 2] = self.idxs[i + 1] * k

        self.marginal_idxs[0] = 1
        for i in range(self.m - 1):
            k = len(np.unique([row[self.m - i - 1] for row in table]))
            self.marginal_idxs[i + 1] = self.marginal_idxs[i] * k

        self.keymap = {tuple(row[:-1]): i for i, row in enumerate(table)}
        self.values = np.array([_log(row[-1]) for row in table], dtype='float64')

        self.marginal_keymap = {tuple(row[:-2]): i for i, row in enumerate(table[::self.k])}
        self.parents = parents
        self.parameters = [table, self.parents]

    def __str__(self):
        return "\n".join("\t".join(map(str, key + (np.exp(self.values[idx]),))) for key, idx in self.keymap.items())

    def __len__(self):
        return self.k

    def keys(self):
        """Return the keys of the probability distribution which has parents, the child variable."""
        return tuple(set(row[-1] for row in self.keymap.keys()))

    def bake(self, keys):
        """Order the inputs according to some external global ordering."""
        keymap, values = [], []
        for i, key in enumerate(keys):
            keymap.append((key, i))
            idx = self.keymap[key]
            values.append(self.values[idx])
        self.marginal_keymap = {tuple(row[:-1]): i for i, row in enumerate(keys[::self.k])}
        for i in range(len(keys)):
            self.values[i] = values[i]
        self.keymap = dict(keymap)

    def sample(self, parent_values=None, n=None, random_state=None):
        """Return a random sample from the conditional probability table."""
        random_state = check_random_state(random_state)
        if parent_values is None:
            parent_values = {}
            for parent in self.parents:
                if parent not in parent_values:
                    parent_values[parent] = parent.sample(random_state=random_state)

        sample_cands, sample_vals = [], []
        for key, ind in self.keymap.items():
            for j, parent in enumerate(self.parents):
                if parent_values[parent] != key[j]:
                    break
            else:
                sample_cands.append(key[-1])
                sample_vals.append(np.exp(self.values[ind]))

        sample_vals /= np.sum(sample_vals)
        if n is None:
            sample_ind = np.where(random_state.multinomial(1, sample_vals))[0][0]
            return sample_cands[sample_ind]
        elif n > 5:
            return random_state.choice(a=sample_cands, p=sample_vals, size=n)
        else:
            states = random_state.randint(1000000, size=n)
            return [self.sample(parent_values, n=None, random_state=state) for state in states]

    def log_probability(self, X):
        """Return the log probability of a value, which is a tuple in proper ordering, like the training data."""
        X = np.array(X, ndmin=2, dtype=object)
        log_probabilities = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            x = tuple(x)
            for x_ in x:
                if _check_nan(x_):
                    break
            else:
                idx = self.keymap[x]
                log_probabilities[i] = self.values[idx]
        if X.shape[0] == 1:
            return log_probabilities[0]
        return log_probabilities

    def joint(self, neighbor_values=None):
        """This will turn a conditional probability table into a joint probability table."""
        neighbor_values = neighbor_values or self.parents + [None]
        if isinstance(neighbor_values, dict):
            neighbor_values = [neighbor_values.get(p, None) for p in self.parents + [self]]

        table, total = [], 0
        for key, idx in self.keymap.items():
            scaled_val = self.values[idx]
            for j, k in enumerate(key):
                if neighbor_values[j] is not None:
                    scaled_val += neighbor_values[j].log_probability(k)
            scaled_val = np.exp(scaled_val)
            total += scaled_val
            table.append(key + (scaled_val,))
        table = [row[:-1] + (row[-1] / total if total > 0 else 1. / self.n,) for row in table]
        return JointProbabilityTable(table, self.parents)

    def marginal(self, neighbor_values=None):
        """Calculate the marginal of the CPT."""
        if isinstance(neighbor_values, dict):
            neighbor_values = [neighbor_values.get(d, None) for d in self.parents]
        i = -1 if neighbor_values is None else neighbor_values.index(None)
        return self.joint(neighbor_values).marginal(i)

    def fit(self, items, weights=None, inertia=0.0, pseudocount=0.0):
        """Update the parameters of the table based on the data."""
        self.summarize(items, weights)
        self.from_summaries(inertia, pseudocount)

    def summarize(self, items, weights=None):
        """Summarize the data into sufficient statistics to store."""
        if len(items) == 0 or self.frozen:
            return
        if weights is None:
            weights = np.ones(len(items), dtype='float64')
        elif np.sum(weights) == 0:
            return
        else:
            weights = np.asarray(weights, dtype='float64')
        self.__summarize(items, weights)

    def __summarize(self, items, weights):
        for i, item in enumerate(items):
            item = tuple(item)
            for symbol in item:
                if _check_nan(symbol):
                    break
            else:
                key = self.keymap[item]
                self.counts[key] += weights[i]
                key = self.marginal_keymap[item[:-1]]
                self.marginal_counts[key] += weights[i]

    def from_summaries(self, inertia=0.0, pseudocount=0.0):
        """Update the parameters of the distribution using sufficient statistics."""
        w_sum = sum(self.counts)
        if w_sum < 1e-7:
            return
        for i in range(self.n):
            k = i // self.k
            if self.marginal_counts[k] > 0:
                probability = ((self.counts[i] + pseudocount) /
                               (self.marginal_counts[k] + pseudocount * self.k))
                self.values[i] = _log(np.exp(self.values[i]) * inertia + probability * (1 - inertia))
            else:
                self.values[i] = -_log(self.k)
        for i in range(self.n):
            idx = self.keymap[tuple(self.parameters[0][i][:-1])]
            self.parameters[0][i][-1] = np.exp(self.values[idx])
        self.clear_summaries()

    def clear_summaries(self):
        """Clear the summary statistics stored in the object."""
        self.counts.fill(0)
        self.marginal_counts.fill(0)

    def to_dict(self):
        table = [list(key + tuple([np.exp(self.values[i])])) for key, i in self.keymap.items()]
        table = [[str(item) for item in row] for row in table]
        return {
            'class': 'Distribution',
            'name': 'ConditionalProbabilityTable',
            'table': table,
            'dtypes': self.dtypes,
            'parents': [dist.to_dict() for dist in self.parents]
        }

    @classmethod
    def from_samples(cls, X, parents=None, weights=None, pseudocount=0.0, keys=None):
        """Learn the table from data."""
        X = np.asarray(X)
        n, d = X.shape
        keys = keys or [np.unique(X[:, i]) for i in range(d)]
        for i in range(d):
            keys_ = [key for key in keys[i] if not _check_nan(key)]
            keys[i] = keys_
        table = [list(key) + [1. / len(keys[-1]), ] for key in it.product(*keys)]
        d = ConditionalProbabilityTable(table, parents)
        d.fit(X, weights, pseudocount=pseudocount)
        return d