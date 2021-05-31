from .kfac import KFACOptimizer
from .ekfac import EKFACOptimizer
from .kbfgs import KBFGSOptimizer
from .kbfgsl import KBFGSLOptimizer
from .kbfgsl_2loop import KBFGSL2LOOPOptimizer
from .kbfgsl_mem_eff import KBFGSLMEOptimizer
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
    elif name == 'kbfgsl_2loop':
        return KBFGSL2LOOPOptimizer
    elif name == 'kbfgsl_mem_eff':
        return KBFGSLMEOptimizer
    elif name == 'ngd':
        return NGDOptimizer
    else:
        raise NotImplementedError