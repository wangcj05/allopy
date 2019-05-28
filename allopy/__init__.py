from allopy.optimize import ASROptimizer, BaseOptimizer, OptData
from ._version import get_versions

v = get_versions()
__version__ = v.get('closest-tag', v['version'])

del v, get_versions
