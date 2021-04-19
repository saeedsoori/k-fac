from .kfac import KFACOptimizer
from .ekfac import EKFACOptimizer
from .kbfgs import KBFGSOptimizer
from .kbfgsl import KBFGSLOptimizer
from .ngd import NGDOptimizer


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'ekfac':
        return EKFACOptimizer
    elif name == 'kbfgs':
    	return KBFGSOptimizer
    elif name == 'kbfgsl':
    	return KBFGSLOptimizer
    elif name == 'ngd':
        return NGDOptimizer
    else:
        raise NotImplementedError