'''
Defines :py:class:`Layer` class.
'''

from pyapprox.sciml.integraloperators import IntegralOperator


class Layer():
    '''
    Layer class. This allows each layer to have potentially multiple integral
    operators, each receiving the output of the previous layer (skip-forward).
    '''
    def __init__(self, integralops):
        '''
        Parameters
        ----------

        integralops : :py:class:`pyapprox.sciml.integraloperators.IntegralOperator`
        or list thereof
            Integral operators that define the Layer instance
        '''
        if (isinstance(integralops, list) and
            all(issubclass(op.__class__, IntegralOperator)
                for op in integralops)):
            self._integralops = integralops
        elif issubclass(integralops.__class__, IntegralOperator):
            self._integralops = [integralops]
        else:
            raise ValueError(
                'integralops must be IntegralOperator, or list '
                'thereof')
        self._hyp_list = sum([op._hyp_list for op in self._integralops])

    def _combine(self, v1, v2):
        """
        Combine two outputs. The default is addition.
        """
        # must use += otherwise gradient cannot be computed correctly
        v1 += v2
        return v1

    def __call__(self, samples):
        r'''
        For layer k, computes

            u_{k+1} = \sum_{i_k=1}^{N_k} integralops[i_k](y_k)
        '''
        layer_output = self._integralops[0](samples)
        for ii in range(1, len(self._integralops)):
            layer_output += self._integralops[ii](samples)
            # layer_output = self._combine(
            #     layer_output, self._integralops[ii](samples))
        return layer_output

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self._hyp_list._short_repr())
