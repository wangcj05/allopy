import pickle
import warnings
from copy import deepcopy
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from copulae.core import is_psd, near_psd
from copulae.types import Array
from muarch.calibrate import calibrate_data
from muarch.funcs import get_annualized_kurtosis, get_annualized_mean, get_annualized_sd, get_annualized_skew

__all__ = ['OptData', 'alter_frequency', 'calibrate_data', 'coalesce_covariance_matrix', 'translate_frequency']


# noinspection PyMissingConstructor
class OptData(np.ndarray):
    """
    Returns an OptData class which is an enhancement of ndarray with helper methods. Most of the methods that can be
    applied to numpy arrays can be applied to an instance of this object too. By default, any index
    """

    def __new__(cls, data: np.ndarray, time_unit='monthly'):
        obj = np.asarray(data).view(cls)
        return obj

    def __init__(self, data: np.ndarray, time_unit='monthly'):
        """
        Returns an OptData class which is an enhancement of ndarray with helper methods

        Parameters
        ----------
        data: ndarray
            3D tensor where the first axis represents the time period, the second axis represents the trials
            and the third axis represents the assets

        time_unit: {int, 'monthly', 'quarterly', 'semi-annually', 'yearly'}, optional
            Specifies how many units (first axis) is required to represent a year. For example, if each time period
            represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
            a month. Alternatively, specify one of 'monthly', 'quarterly', 'semi-annually' or 'yearly'
        """

        assert data.ndim == 3, "Data must be 3 dimensional with shape like (t, n, a) where `t` represents the time " \
                               "periods, `n` represents the trials and `a` represents the assets"

        periods, trials, n_assets = data.shape
        self.time_unit = translate_frequency(time_unit)

        # empirical covariance taken along the time-asset axis then averaged by trials
        # annualized data
        a = (data + 1).reshape(periods // self.time_unit, self.time_unit, trials, n_assets).prod(1) - 1
        cov_mat = np.mean([np.cov(a[i].T) for i in range(periods // self.time_unit)], 0)
        cov_mat = near_psd(cov_mat)
        assert is_psd(cov_mat), "covariance matrix must be positive semi-definite"

        if np.allclose(np.diag(cov_mat), 1) and np.alltrue(np.abs(cov_mat) <= 1):
            warnings.warn("The covariance matrix feels like a correlation matrix. Are you sure it's correct?")

        self.n_years = periods / self.time_unit
        self.n_assets = n_assets

        # minor optimization when using rebalanced optimization. This is essentially a cache
        self._unrebalanced_returns_data: Optional[np.ndarray] = None
        self._cov_mat = cov_mat

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._cov_mat = getattr(obj, 'cov_mat', None)
        self.time_unit = getattr(obj, 'time_unit', 12)
        self.n_years = getattr(obj, 'n_years', 0)
        self.n_assets = getattr(obj, 'n_assets', 0)
        self._unrebalanced_returns_data = getattr(obj, '_unrebalanced_returns_data', None)

    def aggregate_assets(self, w: Iterable[float], columns: Optional[Iterable[float]] = None, copy=True):
        """
        Aggregates the asset columns with the weights given

        The column index (3rd axis) specifies which columns to aggregate. The aggregated column will be the first
        column.

        Parameters
        ----------
        w: iterable float
            The weights to aggregate the columns by. Weights do not have to sum to 1, if it needs to, you should check
            it prior

        columns: iterable int, optional
            The column index of the aggregated data. If not specified, method will aggregate the first :code:`n` columns
            where :math:`n` is the length of :code:`w`
        copy : boolean, optional
            If True, returns a copy of the :class:`OptData`. If False, returns a slice of the original
            :class:`OptData`. This means any change made to the :class:`OptData` will affect the original
            :class:`OptData` instance as it is not a copy.

        Returns
        -------
        OptData
            An instance of :class:`OptData` with the aggregated columns

        Examples
        --------
        If we have a (60 x 1000 x 10) data and we want to aggregate the assets the first 3 indexes,

        >>> from allopy import OptData
        >>> import numpy as np
        >>>
        >>> np.random.seed(8888)
        >>> data = OptData(np.random.standard_normal((60, 1000, 10)))
        >>> data.aggregate_assets([0.3, 0.4, 0.3]).shape
        >>>
        >>> # Alternatively, we can specify the indices directly. Let's assume we want to aggregate indices [4, 1, 6]
        >>> data = OptData(np.random.standard_normal((60, 1000, 10)))
        >>> data.aggregate_assets([0.3, 0.4, 0.3], [4, 1, 6]).shape
        """
        w = np.asarray(w)

        assert w.ndim == 1 and w.size != 0, "`w` must be a non-empty 1D vector"

        if columns is not None:
            columns = np.asarray(columns)
            assert columns.shape == w.shape, "columns must be a 1D integer vector with the same shape as `w`"
        else:
            columns = np.arange(len(w))

        agg = self[..., columns] @ w
        mask = [i not in columns for i in range(self.n_assets)]  # columns that have not changed

        data = OptData(np.concatenate([agg[..., None], self[..., mask]], axis=2), time_unit=self.time_unit)
        data.cov_mat = coalesce_covariance_matrix(self.cov_mat, w, columns)

        return data.copy() if copy else data

    def alter_frequency(self, to: Union[int, str]):
        """
        Coalesces a the 3D tensor to a lower frequency.

        For example, if we had a 10000 simulations of 10 year, monthly returns for 30 asset classes,
        we would originally have a 120 x 10000 x 30 tensor. If we want to collapse this
        to a quarterly returns tensor, the resulting tensor's shape would be 40 x 10000 x 30

        Note that we can only coalesce data from a higher frequency to lower frequency.

        Parameters
        ----------
        to: {int, 'month', 'quarter', 'year'}, optional
            The targeted frequency. If a string is passed in, it must be one of ('month', 'quarter', 'year').
            If an integer is passed in, this value should be the number of units in a year

        Returns
        -------
        OptData
            A :class:`OptData` with lower frequency

        Example
        -------
        >>> import numpy as np
        >>> from allopy import OptData
        >>>
        >>> np.random.seed(8888)
        >>> data = np.random.standard_normal((120, 10000, 7))
        >>> data = OptData(data)
        >>> new_data = data.alter_frequency('quarter')  # changes to new_data will affect data
        >>> print(new_data.shape)
        >>>
        >>> # making copies, changes to new_data will not affect data
        >>> new_data = data.alter_frequency(4)  # this is equivalent of month to year
        """

        data = alter_frequency(np.asarray(self), self.time_unit, to)
        return OptData(data, to)  # Cast as OptData

    def calibrate_data(self, mean: Optional[Iterable[float]] = None, sd: Optional[Iterable[float]] = None,
                       inplace=False):
        """
        Calibrates the data given the target mean and standard deviation.

        Parameters
        ----------
        mean: iterable float, optional
            the targeted mean vector

        sd: iterable float, optional
            the targeted float vector

        inplace: boolean, optional
            If True, the :class:`OptData` is modified inplace. This means that the underlying :class:`OptData`
            is changed. If False, a new instance of :class:`OptData` is returned

        Returns
        -------
        OptData
            an instance of :class:`OptData`
        """
        if inplace:
            return calibrate_data(self, mean, sd, self.time_unit, True)
        return OptData(calibrate_data(self, mean, sd, self.time_unit), self.time_unit)

    @property
    def cov_mat(self):
        """Returns the covariance matrix of the OptData"""
        return self._cov_mat

    @cov_mat.setter
    def cov_mat(self, cov_mat: np.ndarray):
        cov = np.asarray(cov_mat)
        ideal_shape = (self.n_assets, self.n_assets)

        assert cov.shape == ideal_shape, f"covariance matrix should have shape {ideal_shape}"
        self._cov_mat = cov_mat

    def cut_by_horizon(self, years: float, copy=True):
        """
        Returns the :class:`OptData` with the number of years specified.

        For example, given that you have a (120 x ...) :class:`OptData` and each time unit represents a month.
        If the first 5 years is required, this method will return a (60 x ...) :class:`OptData`.

        Parameters
        ----------
        years: float
            number of years

        copy: boolean, optional
            If True, returns a copy of the :class:`OptData`. If False, returns a slice of the original :class:`OptData`.
             This means any change made to the :class:`OptData` will affect the original :class:`OptData` instance
             as it is not a copy.

        Returns
        -------
        OptData
            A new instance of the cut :class:`OptData`
        """
        limit = int(years * self.time_unit)
        data = deepcopy(self)[:limit] if copy else self[:limit]
        data.n_years = years
        return data

    def cvar(self, w: Array, rebalance: bool, percentile=5.0):
        r"""
        Calculates the CVaR given the weights.

        CVaR is calculated as the mean of returns that is below the percentile specified. Technically, it is the
        expected shortfall. The CVaR is given by:

        .. math::

            ES_\alpha = \frac{\sum_{i \in \mathbf{r}} \mathbb{1}_\alpha (r_i) \cdot r_i}
            {\sum_{i \in \mathbf{r}} \mathbb{1}_\alpha (r_i)}

        where :math:`\alpha` is the percentile cutoff, :math:`\mathbf{r}` is the geometric returns vector and
        :math:`\mathbb{1}_\alpha` is an indicator function specifying if the returns instance, :math:`r_i` is below
        the :math:`alpha` cutoff

        Parameters
        ----------
        w: {iterable float, ndarray}
            Portfolio weights

        rebalance: bool
            Whether portfolio is rebalanced every time period

        percentile: float, default 5.0
            The percentile to cutoff for CVaR calculation

        Returns
        -------
        float
            CVaR of the portfolio

        See Also
        --------
        :py:meth:`.portfolio_returns` : Portfolio returns
        """
        assert 0 <= percentile <= 100, "Percentile must be a number between [0, 100]"
        w = _format_weights(w, self)

        returns = self.portfolio_returns(w, rebalance)
        cutoff = np.percentile(returns, percentile)
        return float(returns[returns <= cutoff].mean())

    def expected_return(self, w: Array, rebalance: bool):
        r"""
        Calculates the annualized expected return given a weight vector

        The expected annualized returns is given by

        .. math::

            \mu_R = \frac{1}{N} \sum^N_i {r_i^{1/y} - 1}

        where :math:`r` is an instance of the geometric returns vector and :math:`y` is the number of years.

        Parameters
        ----------
        w: {iterable float, ndarray}
            Portfolio weights

        rebalance: bool
            Whether portfolio is rebalanced every time period

        Returns
        -------
        float
            Annualized return

        See Also
        --------
        :py:meth:`.portfolio_returns` : Portfolio returns

        """
        w = _format_weights(w, self)
        returns = self.portfolio_returns(w, rebalance) + 1

        return (np.sign(returns) * np.abs(returns) ** (1 / self.n_years)).mean() - 1

    def sharpe_ratio(self, w: Array, rebalance: bool) -> float:
        r"""
        Calculates the portfolio sharpe ratio.

        The formula for the sharpe ratio is given by:

        .. math::

            SR = \frac{\mu_R}{\sigma_R}

        Parameters
        ----------
        w: {iterable float, ndarray}
            Portfolio weights

        rebalance: bool
            Whether portfolio is rebalanced every time period

        Returns
        -------
        float
            Portfolio sharpe ratio

        See Also
        --------
        :py:meth:`.expected_return` : Expected returns
        :py:meth:`.portfolio_returns` : Portfolio returns
        :py:meth:`.volatility` : Volatility

        """
        w = _format_weights(w, self)
        e = 1e6 * self.expected_return(w, rebalance)  # added scale for numerical stability during optimization
        v = 1e6 * self.volatility(w)

        return e / v

    def volatility(self, w: Array) -> float:
        r"""
        Calculates the volatility of the portfolio given a weight vector. The volatility is given by:

        .. math::

            \mathbf{w} \cdot \Sigma \cdot \mathbf{w^T}

        where :math:`\mathbf{w}` is the weight vector and :math:`\Sigma` is the asset covariance matrix

        Parameters
        ----------
        w: {iterable float, ndarray}
            Portfolio weights

        Returns
        -------
        float
            Portfolio volatility
        """
        w = _format_weights(w, self)

        return float(w.T @ self.cov_mat @ w) ** 0.5

    def portfolio_returns(self, w: Array, rebalance: bool) -> np.ndarray:
        r"""
        Calculates the vector of geometric returns of the portfolio for every trial in the simulation.

        The simulated returns is a 3D tensor. If there is rebalancing, then the geometric returns for each trial
        is given by:

        .. math::

            r_i = \prod^T (\mathbf{R_i} \cdot \mathbf{w} + 1)  \forall i \in \{ 1, \dots, N \}

        Otherwise, if there is no rebalancing:

        .. math::

            r_i = (\prod^T (\mathbf{R_i} + 1) - 1) \cdot \mathbf{w}  \forall i \in \{ 1, \dots, N \}

        where :math:`r_i` is the geometric returns for trial :math:`i`, :math:`T` is the total time period,
        :math:`\mathbf{R_i}` is the returns matrix for trial :math:`i`, :math:`\mathbf{w}` is the weights vector
        and :math:`N` is the total number of trials.

        Parameters
        ----------
        w: array_like
            Portfolio weights

        rebalance: bool
            Whether portfolio is rebalanced every time period

        Returns
        -------
        ndarray
            vector of portfolio returns
        """
        if rebalance:
            return (self @ w + 1).prod(0) - 1
        else:
            if self._unrebalanced_returns_data is None:  # cache this calculation
                self._unrebalanced_returns_data = np.asarray((self + 1).prod(0) - 1)
            return self._unrebalanced_returns_data @ w

    def set_cov_mat(self, cov_mat: np.ndarray):
        """
        Sets the covariance matrix

        Parameters
        ----------
        cov_mat: ndarray
            Asset covariance matrix

        Returns
        -------
        OptData
            Own OptData instance
        """
        cov = np.asarray(cov_mat)
        ideal_shape = (self.n_assets, self.n_assets)

        assert cov.shape == ideal_shape, f"covariance matrix should have shape {ideal_shape}"
        self._cov_mat = cov_mat
        return self

    @property
    def statistics(self):
        """
        Returns the statistics (4 moments) of the cube

        Returns
        -------
        DataFrame
            The first 4 moments of the cube for each asset (last axis)
        """
        return pd.DataFrame({
            "Mean": get_annualized_mean(self, self.time_unit),
            "SD": get_annualized_sd(self, self.time_unit),
            "Skew": get_annualized_skew(self, self.time_unit),
            "Kurt": get_annualized_kurtosis(self, self.time_unit),
        })

    def take_assets(self, start: int, stop: Optional[int] = None):
        """
        Returns a new :code:`OptData` instance from the specified start and stop index

        Parameters
        ----------
        start: int
            Starting index. If the stop index is not specified, the start index will be 0 and this value will become
            the stop index. Akin to the :code:`range` function.

        stop: int
            Stopping index

        Returns
        -------
        OptData
            A new OptData instance.
        """

        if stop is None:
            start, stop = 0, start

        assert isinstance(start, int) and isinstance(stop, int), "Indices must be integers"
        assert start < stop, "Start index must be less or equal to stop index"

        if start == stop:
            stop += 1

        data: OptData = deepcopy(self)
        data = data[..., start:stop]
        data.n_assets = stop - start
        data.cov_mat = data.cov_mat[start:stop, start:stop]

        return data

    def to_pickle(self, path: str):
        """
        Saves the OptData object as a pickle file

        Parameters
        ----------
        path: str
            file path of the pickle file
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Overwrote the default :code:`ufunc` function so that we can get :class:`OptData` if calculated data is
        3 dimensional, :class:`float` if calculated data has 0 dimension or is 1D and len 1 and :class:`ndarray`
        otherwise (2D or >= 4D).

        Parameters
        ----------
        ufunc:
            The :code:`ufunc` object that was called.

        method: { '__call__', 'reduce', 'reduceat', 'accumulate', 'outer', 'inner' }
            A string indicating which :code:`ufunc` method was called

        inputs: tuple
            tuple of the input arguments to the :code:`ufunc`.

        kwargs: keyword arguments
            is a dictionary containing the optional input arguments of the :code:`ufunc`.. If given, any out arguments,
            both positional and keyword, are passed as a tuple in kwargs

        Returns
        -------
        {float, ndarray, OptData}
            Depending on the shape of calculated data, will return 1 of float, ndarray or OptData
        """
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, OptData):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, OptData):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super(OptData, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], OptData):
                inputs[0].info = info
            return
        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(OptData)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        results = results[0] if len(results) == 1 else results

        if isinstance(results, OptData):
            if results.ndim in (0, 1) or len(results) == 1:
                return float(results)
            if results.ndim != 3:
                return np.asarray(results)
        return results

    def __reduce__(self):
        """
        This function is used to formulate data that will be sent for pickling.

        The end result is the actual blob that will be pickled. Added the class properties explicitly as these are not
        passed into pickle by default

        Returns
        -------
        tuple
            Tuple object containing data to be sent for pickling
        """
        *state, meta = super().__reduce__()
        meta = meta + ({
                           'cov_mat': self._cov_mat,
                           'n_years': self.n_years,
                           'n_assets': self.n_assets,
                           'time_unit': self.time_unit,
                           '_unrebalanced_returns_data': self._unrebalanced_returns_data
                       },)
        return (*state, meta)

    def __setstate__(self, state, *args, **kwargs):
        """
        This function is used to recover the class instance from the pickle object. It is called by pickle by default.

        Parameters
        ----------
        state: tuple of objects
            This state provided is the primary data that will be used to recover the class object

        args
            arguments

        kwargs
            keyword arguments
        """
        meta = state[-1]
        self.n_assets = meta['n_assets']
        self.cov_mat = meta['cov_mat']
        self.n_years = meta['n_years']
        self.time_unit = meta['time_unit']
        self._unrebalanced_returns_data = meta['_unrebalanced_returns_data']

        super(OptData, self).__setstate__(state[:-1], *args, **kwargs)


def alter_frequency(data, from_='month', to_='quarter'):
    """
    Coalesces a the 3D tensor to a lower frequency.

    For example, if we had a 10000 simulations of 10 year, monthly returns for 30 asset classes,
    we would originally have a 120 x 10000 x 30 tensor. If we want to collapse this
    to a quarterly returns tensor, the resulting tensor's shape would be 40 x 10000 x 30

    Note that we can only coalesce data from a higher frequency to lower frequency.

    Parameters
    ----------
    data: ndarray
        The 3-dimension simulation tensor. The data's dimensions must be in time, trials, asset.

    from_: {int, 'month', 'quarter', 'year'}, optional
        The starting frequency. If a string is passed in, it must be one of ('month', 'quarter', 'year').
        If an integer is passed in, this value should be the number of units in a year. Thus, if moving
        from monthly data to quarterly data, this argument should be 12

    to_: {int, 'month', 'quarter', 'year'}, optional
        The targeted frequency. If a string is passed in, it must be one of ('month', 'quarter', 'year').
        If an integer is passed in, this value should be the number of units in a year. Thus, if moving
        from monthly data to quarterly data, this argument should be 4

    Returns
    -------
    OptData
        A :class:`OptData` with lower frequency

    Example
    -------
    >>> import numpy as np
    >>> from allopy.opt_data import alter_frequency
    >>>
    >>> np.random.seed(8888)
    >>> data = np.random.standard_normal((120, 10000, 7))
    >>> new_data = alter_frequency(data, 'month', 'quarter')
    >>> print(new_data.shape)
    >>>
    >>> # making copies, changes to new_data will not affect data
    >>> new_data = alter_frequency(data, 12, 4)  # this is equivalent of month to quarter
    """

    # type check and convert strings to integers
    to_ = translate_frequency(to_)
    from_ = translate_frequency(from_)

    if to_ == from_:
        return data

    assert from_ > to_, "Cannot extend data from lower to higher frequency. For example, we " \
                        "cannot go from yearly data to monthly data. How to fill anything in between?"

    t, n, s = data.shape
    new_t = t / from_ * to_

    assert new_t.is_integer(), f"cannot convert {t} periods to {new_t} periods. Targeted periods must be an integer"
    new_t = int(new_t)

    return (data.reshape((new_t, t // new_t, n, s)) + 1).prod(1) - 1  # reshape data


def coalesce_covariance_matrix(cov,
                               w: Iterable[float],
                               indices: Optional[Iterable[int]] = None) -> Union[np.ndarray, float]:
    """
    Aggregates the covariance with the weights given at the indices specified
    The aggregated column will be the first column.
    Parameters
    ----------
    cov: ndarray
        Covariance matrix of the portfolio
    w: ndarray
        The weights to aggregate the columns by. Weights do not have to sum to 1, if it needs to, you should check
        it prior
    indices: iterable int, optional
        The column index of the aggregated data. If not specified, method will aggregate the first 'n' columns
        where 'n' is the length of :code:`w`
    Returns
    -------
    ndarray
        Aggregated covariance matrix
    Examples
    --------
    If we have a (60 x 1000 x 10) data and we want to aggregate the assets the first 3 indexes,
    >>> from allopy.opt_data import coalesce_covariance_matrix
    >>> import numpy as np
    form covariance matrix
    >>> np.random.seed(8888)
    >>> cov = np.random.standard_normal((5, 5))
    >>> cov = cov @ cov.T
    coalesce first and second column where contribution is (30%, 70%) respectively.
    Does not have to sum to 1
    >>> coalesce_covariance_matrix(cov, (0.3, 0.7))
    coalesce fourth and fifth column
    >>> coalesce_covariance_matrix(cov, (0.2, 0.4), (3, 4))
    """
    w = np.asarray(w)
    cov = np.asarray(cov)
    n = len(w)

    assert cov.ndim == 2 and cov.shape[0] == cov.shape[1], 'cov must be a square matrix'
    assert n <= len(cov), 'adjustment weights cannot be larger than the covariance matrix'

    if indices is None:
        indices = np.arange(n)

    _, a = cov.shape  # get number of assets originally

    # form transform matrix
    T = np.zeros((a - n + 1, a))
    T[0, :n] = w
    T[1:, n:] = np.eye(a - n)

    # re-order covariance matrix
    rev_indices = sorted(set(range(a)) - set(indices))  # these are the indices that are not aggregated
    indices = [*indices, *rev_indices]
    cov = cov[indices][:, indices]  # reorder the covariance matrix, first by rows then by columns

    cov = T @ cov @ T.T
    return float(cov) if cov.size == 1 else near_psd(cov)


def translate_frequency(_freq: Union[str, int]) -> int:
    """Translates a given frequency to the integer equivalent with checks"""
    if isinstance(_freq, str):
        _freq_ = _freq.lower()
        if _freq_ in ('m', 'month', 'monthly'):
            return 12
        elif _freq_ in ('s', 'semi-annual', 'semi-annually'):
            return 6
        elif _freq_ in ('q', 'quarter', 'quarterly'):
            return 4
        elif _freq_ in ('y', 'a', 'year', 'annual', 'yearly', 'annually'):  # year
            return 1
        else:
            raise ValueError(f'unknown frequency {_freq}. Use one of month, semi-annual, quarter or annual')

    assert isinstance(_freq, int) and _freq > 0, 'frequency can only be a positive integer or a string name'
    return _freq


def _format_weights(w, data: OptData) -> np.ndarray:
    """Formats weight inputs. Raises errors if the weights do not have the right number of elements"""
    w = np.ravel(w)
    assert len(w) == data.n_assets, f'input weights should have {data.n_assets} elements'
    return w
