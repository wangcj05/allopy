import pandas as pd
from os import path

__data_dir = path.join(path.dirname(__file__), '..', 'data')


def read_analytics():
    fp = path.join(__data_dir, 'analytics.xlsx')
    attr = pd.read_excel(fp, 'attr')
    _crp = pd.read_excel(fp, 'crp', header=[0, 1])
    crp = {'rebal': _crp['rebal'], 'no_rebal': _crp['no_rebal']}
    metrics = pd.read_excel(fp, 'metrics', index_col=0, header=[0, 1])

    return {
        'attr': attr,
        'crp': crp,
        'metrics': metrics
    }


def read_historical(typ='total', sheet=None) -> pd.DataFrame:
    typ = typ.lower()
    valid_types = 'total', 'passive', 'active', 'expected'
    if typ not in valid_types:
        raise ValueError(f"Unrecognized type: {typ}. It must be one of {valid_types}")

    fp = path.join(__data_dir, 'historical.xlsx')

    if typ == 'expected':
        if sheet is None:
            raise ValueError("<sheet> must be defined if <typ> is 'expected'")
        return pd.read_excel(fp, sheet)

    hist = pd.read_excel(fp, 'data', parse_dates=['DATE'])

    if typ == 'passive':
        return hist.iloc[:, :8]
    else:
        return hist[hist.DATE >= '31-01-2007']
