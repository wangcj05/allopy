import numpy as np
import scipy.optimize as opt
import warnings
from copulae.core import is_psd, near_psd
from copulae.types import Array
from functools import wraps
from typing import Iterable, Optional, Union

from allopy.analytics import annualize_returns, coalesce_covariance_matrix


def __format_weights__(func):
    """
    Formats weight inputs. Raises errors if the weights do not have the right number of elements

    :param func: callable
        methods where the first input is the ideal weights
    """

    @wraps(func)
    def decorator(self: 'OptData', w: Array, *args, **kwargs):
        w = np.ravel(w)

        if len(w) != self.n_assets:
            raise ValueError(f'input weights should have {self.n_assets} elements')

        return func(self, w, *args, **kwargs)

    return decorator


# noinspection PyMissingConstructor
class OptData(np.ndarray):
    """
    Returns an OptData class which is an enhancement of ndarray with helper methods. Most of the methods that can be
    applied to numpy arrays can be applied to an instance of this object too. By default, any index
    """

    def __new__(cls, data: np.ndarray, cov_mat: Optional[np.ndarray] = None, time_unit=12):
        obj = np.asarray(data).view(cls)
        return obj

    def __init__(self, data: np.ndarray, cov_mat: Optional[np.ndarray] = None, time_unit=12):
        """
        Returns an OptData class which is an enhancement of ndarray with helper methods

        Parameters
        ----------
        data: ndarray
            3D tensor where the first axis represents the time period, the second axis represents the trials
            and the third axis represents the assets

        cov_mat: ndarray, optional
            2D matrix which describes the covariance among the asset classes. If it is not specified, it will default to
            the sample covariance from the data tensor

        time_unit: int, optional
            Specifies how many units (first axis) is required to represent a year. For example, if each time period
            represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
            a month
        """

        if data.ndim != 3:
            raise ValueError("Data must be 3 dimensional with shape like (t, n, a) where <t> represents the time "
                             "periods, <n> represents the trials and <a> represents the assets")

        periods, trials, n_assets = data.shape
        if cov_mat is None:
            # empirical covariance taken along the time-asset axis then averaged by trials
            cov_mat = np.mean([np.cov(data[:, i, :].T) for i in range(trials)], 0)
            cov_mat = near_psd(cov_mat)

        if not is_psd(cov_mat):
            raise ValueError("covariance matrix must be positive semi-definite")

        if np.allclose(np.diag(cov_mat), 1) and np.alltrue(np.abs(cov_mat) <= 1):
            warnings.warn("The covariance matrix feels like a correlation matrix. Are you sure it's correct?")

        self.cov_mat = cov_mat
        self.time_unit = time_unit
        self.n_years = periods / self.time_unit
        self.n_assets = n_assets

        # minor optimization when using rebalanced optimization. This is essentially a cache
        self._unrebalanced_returns_data: np.ndarray = None

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.cov_mat = getattr(obj, 'cov_mat', None)
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

        >>> from allopy.optimize import OptData
        >>> import numpy as np

        >>> np.random.seed(8888)
        >>> data = OptData(np.random.standard_normal((60, 1000, 10)))
        >>> data.aggregate_assets([0.3, 0.4, 0.3]).shape

        Alternatively, we can specify the indices directly. Let's assume we want to aggregate indices [4, 1, 6]
        >>> data = OptData(np.random.standard_normal((60, 1000, 10)))
        >>> data.aggregate_assets([0.3, 0.4, 0.3], [4, 1, 6]).shape
        """
        w = np.asarray(w)
        if w.ndim != 1:
            raise ValueError("<w> must be a 1D float vector")
        elif len(w) == 0:
            raise ValueError("<w> must not be empty")

        if columns is not None:
            columns = np.asarray(columns)
            if columns.ndim != 1 or len(columns) != len(w):
                raise ValueError("columns must be a 1D integer vector with the same length as <w>")
        else:
            columns = np.arange(len(w))

        agg = self[..., columns] @ w
        mask = [i not in columns for i in range(self.n_assets)]  # columns that have not changed

        cov = coalesce_covariance_matrix(self.cov_mat, w, columns)
        data = OptData(np.concatenate([agg[..., None], self[..., mask]], axis=2), cov, time_unit=self.time_unit)

        return data.copy() if copy else data

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
        data = self if inplace else self.copy()
        if sd is not None:
            _calibrate_sd(data, sd)

        if mean is not None:
            _calibrate_mean(data, mean)

        return data

    def coalesce_frequency(self, from_='month', to_='quarter'):
        """
        Coalesces a the 3D tensor to a lower frequency.

        For example, if we had a 10000 simulations of 10 year, monthly
        returns for 30 asset classes, we would originally have a 120 x 10000 x 30 tensor. If we want to collapse this
        to a quarterly returns tensor, the resulting tensor's shape would be 40 x 10000 x 30

        Parameters
        ----------
        from_: {int, 'month', 'quarter', 'year'}, optional
            The frequency of the underlying. If a string is passed in, it must be one of ('month', 'quarter', 'year').
            If an integer is passed in, this value should be the number of units in a year

        to_: {int, 'month', 'quarter', 'year'}, optional
            The targeted frequency. If a string is passed in, it must be one of ('month', 'quarter', 'year').
            If an integer is passed in, this value should be the number of units in a year

        Returns
        -------
        OptData
            A :class:`OptData` with lower frequency

        Example
        -------
        >>> import numpy as np
        >>> from allopy.optimize import OptData

        >>> np.random.seed(8888)
        >>> data = np.random.standard_normal((120, 10000, 7))
        >>> data = OptData(data)
        >>> new_data = data.coalesce_frequency('month', 'quarter')  # changes to new_data will affect data
        >>> print(new_data.shape)

        # making copies, changes to new_data will not affect data
        >>> new_data = data.coalesce_frequency(12, 4, copy=True)  # this is equivalent of month to year
        """

        if type(from_) != type(to_):
            raise ValueError

        # type check and convert strings to integers
        from_ = _freq_convert(from_, 'from_')
        to_ = _freq_convert(to_, 'to_')

        if from_ == to_:
            return self
        elif from_ < to_:
            raise ValueError("Cannot extend data from a shorter time period to a longer time period. For example, we "
                             "cannot go from yearly data to monthly data. How to fill anything in between?")

        t, n, s = self.shape
        new_t = t / from_ * to_
        if not new_t.is_integer():
            raise ValueError(f"cannot convert {t} periods to {new_t} periods. Targeted periods must be an integer")
        new_t = int(new_t)

        data = (self.reshape((new_t, t // new_t, n, s)) + 1).prod(1) - 1  # reshape data
        return OptData(data, self.cov_mat, self.time_unit * to_ // from_)  # Cast as OptData

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
        data = self.copy()[:limit] if copy else self[:limit]
        data.n_years = years
        return data

    @__format_weights__
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
        if percentile < 0 or percentile > 100:
            raise ValueError("Percentile must be a number between [0, 100]")

        returns = self.portfolio_returns(w, rebalance)
        cutoff = np.percentile(returns, percentile)
        return float(returns[returns <= cutoff].mean())

    @__format_weights__
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
        returns = self.portfolio_returns(w, rebalance) + 1
        return annualize_returns(returns, self.n_years).mean()

    @__format_weights__
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
        e = 1e6 * self.expected_return(w, rebalance)  # added scale for numerical stability during optimization
        v = 1e6 * self.volatility(w)
        return e / v

    @__format_weights__
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
        w: {iterable float, ndarray}
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
                           'cov_mat': self.cov_mat,
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
        self.cov_mat = meta['cov_mat']
        self.n_years = meta['n_years']
        self.n_assets = meta['n_assets']
        self.time_unit = meta['time_unit']
        self._unrebalanced_returns_data = meta['_unrebalanced_returns_data']

        super(OptData, self).__setstate__(state[:-1], *args, **kwargs)


def _calibrate_sd(data: OptData, sd: Iterable[float]):
    """
    Adjusts the standard deviation of the data in place

    Parameters
    ----------
    data: OptData
        Data to be adjusted

    sd: iterable float
        Target standard deviation

    """
    sd = np.asarray(sd)
    if len(sd) != data.shape[2]:
        raise ValueError(f'targeted standard deviation needs to have {data.shape[2]} elements')

    scale = sd / data.std(1).mean(0) / np.sqrt(data.time_unit)
    data *= scale


def _calibrate_mean(data: OptData, mean: Iterable[float]):
    """
    Adjusts the mean of the data in place

    Parameters
    ----------
    data: OptData
        Data to be adjusted

    mean: iterable float
        Target mean

    """
    mean = np.asarray(mean)
    if len(mean) != data.shape[2]:
        raise ValueError(f'targeted mean needs to have {data.shape[2]} elements')

    sol = np.array([opt.newton(_geometric_returns,
                               np.random.uniform(-0.02, 0.02),
                               args=(data[..., i], m, data.n_years))
                    for i, m in enumerate(mean)])

    data -= sol[None, None, :]


def _freq_convert(freq: Union[str, int], freq_name: str) -> int:
    known_freq = ('m', 'month', 'q', 'quarter', 'y', 'year')

    if type(freq) is str:
        freq = freq.lower()
        if freq not in known_freq:
            raise ValueError(f"{freq_name} must be in one of {known_freq} or be an integer")

        if freq in ('m', 'month'):
            return 12
        elif freq in ('q', 'quarter'):
            return 4
        else:  # year
            return 1

    if type(freq) is int:
        if freq <= 0:
            raise ValueError(f"{freq_name} must be an integer greater than 0")
        return freq
    else:
        raise ValueError(f"{freq_name} must be in one of {known_freq} or be an integer")


def _geometric_returns(x: float, data: np.ndarray, mean: float, years: float):
    """
    Internal function to adjust the returns matrix. Note that because scipy optimize works on scalar (x), the data
    we pass in has to be a matrix where each matrix represents the time * trials of an asset.

    Parameters
    ----------
    x: float
        The adjustment factor. Put in by :py:func:`scipy.optimize` to determine how much to shift the mean by so
        that we get to 0 for the root finding process

    data: ndarray
        The matrix containing the time and trials

    mean: float
        The target mean value

    years: float
        Number of years. Used for annualizing returns

    Returns
    -------
    float
        The amount to shift the mean by
    """
    cum_returns = (data - x + 1).prod(0)  # shift each asset class by 'x' and take product down time axis
    annualized_mean = np.power(cum_returns, 1 / years).mean()  # annualize the matrix then take mean
    return annualized_mean - mean - 1  # take the annualized minus the targeted 'mean'
