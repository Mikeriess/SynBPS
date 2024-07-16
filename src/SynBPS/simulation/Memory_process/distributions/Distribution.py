import numpy
import sys

#from distributions.DiscreteDistribution import DiscreteDistribution
#from distributions.ConditionalProbabilityTable import ConditionalProbabilityTable
#from distributions.JointProbabilityTable import JointProbabilityTable

def weight_set(items, weights):
	"""Converts both items and weights to appropriate numpy arrays.

	Convert the items into a numpy array with 64-bit floats, and the weight
	array to the same. If no weights are passed in, then return a numpy array
	with uniform weights.
	"""

	items = numpy.array(items, dtype=numpy.float64)
	if weights is None: # Weight everything 1 if no weights specified
		weights = numpy.ones(items.shape[0], dtype=numpy.float64)
	else: # Force whatever we have to be a Numpy array
		weights = numpy.asarray(weights, dtype=numpy.float64)

	return items, weights


class Distribution:
    """A probability distribution.

    Represents a probability distribution over the defined support. This is
    the base class which must be subclassed to specific probability
    distributions. All distributions have the below methods exposed.

    Parameters
    ----------
    Varies on distribution.

    Attributes
    ----------
    name : str
        The name of the type of distribution.
    summaries : list
        Sufficient statistics to store the update.
    frozen : bool
        Whether or not the distribution will be updated during training.
    d : int
        The dimensionality of the data. Univariate distributions are all
        1, while multivariate distributions are > 1.
    """

    def __init__(self):
        self.name = "Distribution"
        self.frozen = False
        self.summaries = []
        self.d = 1

    def marginal(self, *args, **kwargs):
        """Return the marginal of the distribution.

        Parameters
        ----------
        *args : optional
            Arguments to pass in to specific distributions
        **kwargs : optional
            Keyword arguments to pass in to specific distributions

        Returns
        -------
        distribution : Distribution
            The marginal distribution. If this is a multivariate distribution
            then this method is filled in. Otherwise returns self.
        """
        return self

    def copy(self):
        """Return a deep copy of this distribution object.

        This object will not be tied to any other distribution or connected
        in any form.

        Parameters
        ----------
        None

        Returns
        -------
        distribution : Distribution
            A copy of the distribution with the same parameters.
        """
        return self.__class__(*self.parameters)

    def log_probability(self, X):
        """Return the log probability of the given X under this distribution.

        Parameters
        ----------
        X : double
            The X to calculate the log probability of (overridden for
            DiscreteDistributions)

        Returns
        -------
        logp : double
            The log probability of that point under the distribution.
        """
        n = 1 if isinstance(X, (int, float)) else len(X)
        logp_array = numpy.empty(n, dtype='float64')
        X_ndarray = numpy.asarray(X, dtype='float64')

        self._log_probability(X_ndarray, logp_array, n)

        if n == 1:
            return logp_array[0]
        else:
            return logp_array

    @classmethod
    def from_dict(cls, d):
        if ' ' in d['class'] or 'Distribution' not in d['class']:
            raise SyntaxError("Distribution object attempting to read invalid object.")

        elif d['name'] == 'DiscreteDistribution':
            dp = d['parameters'][0]

            if d['dtype'] in ('str', 'unicode', 'numpy.string_'):
                dist = {str(key): value for key, value in dp.items()}
            elif d['dtype'] == 'bool':
                dist = {key == 'True': value for key, value in dp.items()}
            elif d['dtype'] == 'int':
                dist = {int(key): value for key, value in dp.items()}
            elif d['dtype'] == 'float':
                dist = {float(key): value for key, value in dp.items()}
            elif d['dtype'].startswith('numpy.'):
                dtype = d['dtype'][6:]
                dist = {numpy.array([key], dtype=dtype)[0]: value for key, value in dp.items()}
            else:
                dist = dp

            return DiscreteDistribution(dist, frozen=d['frozen'])

        elif 'Table' in d['name']:
            parents = [j if isinstance(j, int) else Distribution.from_dict(j) for j in d['parents']]
            table = []

            for row in d['table']:
                table.append([])
                for dtype, item in zip(d['dtypes'], row):
                    if dtype in ('str', 'unicode', 'numpy.string_'):
                        table[-1].append(str(item))
                    elif dtype == 'bool':
                        table[-1].append(item == 'True')
                    elif dtype == 'int':
                        table[-1].append(int(item))
                    elif dtype == 'float':
                        table[-1].append(float(item))
                    elif dtype.startswith('numpy.'):
                        dtype = dtype[6:]
                        table[-1].append(numpy.array([item], dtype=dtype)[0])
                    else:
                        table[-1].append(item)

            if d['name'] == 'JointProbabilityTable':
                return JointProbabilityTable(table, parents)
            
            elif d['name'] == 'ConditionalProbabilityTable':
                return ConditionalProbabilityTable(table, parents)
            else:
                dist = eval("{}({}, frozen={})".format(d['name'], ','.join(map(str, d['parameters'])), d['frozen']))
                return dist

    @classmethod
    def from_samples(cls, items, weights=None, **kwargs):
        """Fit a distribution to some data without pre-specifying it."""
        distribution = cls.blank()
        distribution.fit(items, weights, **kwargs)
        return distribution

class MultivariateDistribution(Distribution):
    """
    An object to easily identify multivariate distributions such as tables.
    """

    def log_probability(self, X):
        """Return the log probability of the given X under this distribution.

        Parameters
        ----------
        X : list or numpy.ndarray
            The point or points to calculate the log probability of. If one
            point is passed in, then it will return a single log probability.
            If a vector of points is passed in, then it will return a vector
            of log probabilities.

        Returns
        -------
        logp : float or numpy.ndarray
            The log probability of that point under the distribution. If a
            single point is passed in, it will return a single float
            corresponding to that point. If a vector of points is passed in
            then it will return a numpy array of log probabilities for each
            point.
        """

        if isinstance(X[0], (int, float)) or len(X) == 1:
            n = 1
        else:
            n = len(X)

        X_ndarray = np.asarray(X, dtype='float64')

        logp_array = np.empty(n, dtype='float64')

        self._log_probability(X_ndarray, logp_array, n)

        if n == 1:
            return logp_array[0]
        else:
            return logp_array