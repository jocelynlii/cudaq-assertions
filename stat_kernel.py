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
        the expected and observed distributions, and compares them through the chi-square test.

        Args:
            pcrit(float): critical p-value
            expval(int or string or None): the expected value
                If no expected value specified, then this assertion just checks
                that the measurement outcomes are in any classical state.
            negate(bool): True if assertion passed is negation of statistical test passed
            params: params that the kernel needs to take in to run

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
    

    def uniform_assertion(self, pcrit, negate=False, params=[]):
        """
        Performs a chi-squared statistical test on the observed measurement 
        distribution, which is obtained by calling cudaq.sample. Internally, compares
        a normalized table of experimental counts to the scipy.stats.chisquare default, for which
        all outcomes are equally likely.

        Args:
            pcrit(float): critical p-value
            negate(bool): True if assertion passed is negation of statistical test passed
            params: params that the kernel needs to take in to run

        Returns:
            tuple: tuple containing:

                chisq(float): the chi-square value

                pval(float): the p-value

                passed(Boolean): if the test passed
        """
        counts = cudaq.sample(self, *params)

        dict_result = dict(counts.items())

        vals_list = list(dict_result.values()) 
        numqubits = len(str(list(dict_result)[0])) 

        numzeros = 2 ** numqubits - len(dict_result) 

        vals_list.extend([0] * numzeros) 

        chisq, pval = chisquare(vals_list)
        if negate:
            passed = not bool(pval >= pcrit)
        else:
            passed = bool(pval >= pcrit)

        return (chisq, pval, passed)
    
    def product_assertion(self, pcrit, q0len, q1len, negate=False, params=[]):
        """
        Performs a chi-squared contingency test on the observed measurement 
        distribution, which is obtained by calling cudaq.sample.
        Internally, constructs a contingency table from the observed measurement
        distribution counts, then feeds it into scipy.stats.fisher_exact.

        Args:
            pcrit(float): critical p-value
            q0len(int): length (number of qubits) of qubit group 0
            q1len(int): length (number of qubits) of qubit group 1
            negate(bool): True if assertion passed is negation of statistical test passed
            params: params that the kernel needs to take in to run

        Returns:
            tuple: tuple containing:

                odds_ratio(float): the odds ratio, returned from scipy.stats.fisher_exact

                pval(float): the p-value

                passed(bool): if the test passed
        """
        counts = cudaq.sample(self, *params)
        dict_result = dict(counts.items())

        if (len(list(dict_result.keys())[0]) == 1):
            raise Exception("Only 1 qubit -- product state assertion requires at least 2 qubits")

        if (q0len + q1len != len(list(dict_result.keys())[0])):
            raise Exception("2 qubit lengths passed in do not add up to the expected total length")
        
        cont_table = np.zeros((2 ** q0len, 2 ** q1len))

        for (key, value) in dict_result.items():
            q0index = int(key[:q0len], 2) 
            q1index = int(key[q0len:], 2)
            cont_table[q0index][q1index] = value 

        odds_ratio, pval = fisher_exact(cont_table) 

        if negate:
            passed = not bool(pval >= pcrit) 
        else:
            passed = bool(pval >= pcrit)

        return (odds_ratio, pval, passed)

def make_kernel(*args):
    """
        Function used to instantiate a StatKernel instance.
    """
    kernel = StatKernel(*args)
    if len([*args]) == 0:
        return kernel

    return kernel, *kernel.arguments

