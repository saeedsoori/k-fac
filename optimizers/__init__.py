from .kfac import KFACOptimizer
from .ekfac import EKFACOptimizer
from .kbfgs import KBFGSOptimizer


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'ekfac':
        return EKFACOptimizer
    elif name == 'kbfgs':
    	return KBFGSOptimizer
    else:
        raise NotImplementedError