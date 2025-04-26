import cudaq
import numpy as np
from scipy.stats import chisquare
from scipy.stats import fisher_exact

class StatKernel(cudaq.PyKernel):
    """
        Creates a Kernel that includes statistical assertion methods as part of its class definition.
        Inherits from CUDA-Q's PyKernel class.
    """
    def __init__(self, *args):
        """
        Constructor for StatKernel. Runs the same constructor as the one for PyKernel.
        """
        super().__init__([*args])

    def classical_assertion(self, pcrit, expval=None, negate=False, params=[]):
        """
        Performs a chi-squared statistical test on the observed measurement distribution, which
        is obtained by calling cudaq.sample. Internally, it builds an expected distribution that 
        is a table of all 1's exccept for the expected value, which is 2^16. Then, it normalizes 
        the expected and observedn distributions, and compares them through the chi-square test.

        Args:
            pcrit:

        Returns:
            tuple: tuple containing:

                chisq(float): the chi-squared value

                pval(float): the p-value

                passed(bool): if the test passed
        """
        counts = cudaq.sample(self, *params)

        dict_result = dict(counts.items())

        vals_list = list(dict_result.values()) 

        if expval is None:
            index = np.argmax(vals_list) 
        else:
            try:
                index = list(map(lambda x: int (x,2), dict_result.keys())).index(expval)
            except ValueError:
                index = -1

        numqubits = len(str(list(dict_result)[0])) 

        numzeros = 2 ** numqubits - len(dict_result) 
        vals_list.extend([0] * numzeros) 

        exp_list = [1] * len(vals_list) 
        exp_list[index] = 2 ** 16 

        vals_list = vals_list / np.sum(vals_list)
        exp_list = exp_list / np.sum(exp_list)

        chisq, pval = chisquare(vals_list, f_exp=exp_list, ddof=1) 

        if len(str(list(dict_result.keys())[0])) == 1: 
            pval = vals_list[index] 
            passed = bool(pval >= 1 - pcrit) 
        else:
            passed = bool(pval >= pcrit) 
        
        if negate:
            passed = not passed
        else:
            passed = passed

        return (chisq, pval, passed)
