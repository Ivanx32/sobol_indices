import math
import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt

import errors as err
import distribution as distr

import copy
import logging

def get_q_norm(multiIndex, q):
  val = (sum([ind**q for ind in multiIndex]))**(1.0/q)
  return val


def powerset(s):
    els = []
    x = len(s)
    for i in range(1 << x):
        el = [s[j] for j in range(x) if (i & (1 << j))]
        el = tuple(el)
        if len(el)>0:
            els += [el]
    return els


def get_total_based_full(main_full):
    return 


class PCE():
    """
    Ordinary least squares Linear Regression for PCE model.
    """

    def __init__(self, marginals, basis_truncation=None):
        # if basis_truncation is None:
        #     basis_truncation = {'max_degree':1} # , 'q_norm':1.0
        self.basis_truncation = basis_truncation

        if not isinstance(marginals, distr.Marginals):
            raise Exception('Input argument "marginals" must be an instance of Marginals')
        self.marginals = marginals
        self.input_dim = marginals.dim

        self.coef_ = None
        self._basis_functions_number = None
        self._multiIndices = None
        self._model = None
        self._main_sobol_indices = None
        self._total_sobol_indices = None

        self._generate_polynomial_basis()

    def copy(self):
        return copy.deepcopy(self)


    def _generate_polynomial_basis(self):

        if 'multiIndices' in self.basis_truncation.keys():
            multiIndices = self.basis_truncation['multiIndices']
        
        else:

            dim = self.input_dim
            
            if 'max_degree' in self.basis_truncation.keys():
                max_degree = self.basis_truncation['max_degree']
 
                t = [np.array(range(0, max_degree+1))]*dim
                multiIndices = np.array(np.meshgrid(*t)).T.reshape(-1,dim)
                multiIndices = multiIndices.tolist()

            elif 'total_degree' in self.basis_truncation.keys():

                multiIndices = []
                
                total_degree = self.basis_truncation['total_degree']
            
                if 'q_norm' in self.basis_truncation.keys():
                    q_norm = self.basis_truncation['q_norm']
                else:
                    q_norm = None
                
                def fixed_sum_digits(ssum=0, index=[], dim = None, Total = None):
                  #global multiIndices
                  if dim == 0:
                      if ssum == Total:
                        multiIndices.append(index) 
                  else:    
                    for i in range(Total - ssum + 1):
                        fixed_sum_digits(ssum = ssum + i, dim=dim-1, index=index+[i], Total=Total)                 
                        
                for degree in range(total_degree+1):
                    fixed_sum_digits(dim=dim, Total=degree)

                if q_norm is not None:
                    multiIndices = [multiIndex for multiIndex in multiIndices 
                        if get_q_norm(multiIndex, q_norm) <= total_degree]
            
            else: #!!!
                raise Exception('Unknown truncation scheme') #!!!

        self._multiIndices = multiIndices
        self._basis_functions_number = len(multiIndices)
        self._x2fea = lambda X: self._getExtendedDesignMatrix(X)


    def _getExtendedDesignMatrix(self, X):
      
      N, dim = X.shape
      extendedDesignMatrix = np.ones((N, self._basis_functions_number)) # np.nan * 

      for i, multiIndex in enumerate(self._multiIndices):
          P = self.marginals.get_orthonormal_polynomial(multiIndex)
          extendedDesignMatrix[:, i] = P(X).flatten()

      return extendedDesignMatrix


    def set_coef_(self, coeffs):       
        self.coef_ = coeffs
        self._main_sobol_indices = self.get_sobol_indices(ind_type='main')
        self._total_sobol_indices = self.get_sobol_indices(ind_type='total')
        self._model = lambda X: np.matmul(self._getExtendedDesignMatrix(X), self.coef_) #!!!
        
    
    def fit(self, trainX, trainY, method='ls', ls_lambda=0.0, verbose=0):
        self._check_input_array(trainX)
        self._check_train_X_Y(trainX, trainY)
        
        sample_size, dim = trainX.shape 
        Phi = self._getExtendedDesignMatrix(trainX)

        if method == 'ls':
    
            if Phi.shape[0] < Phi.shape[1]:
                raise Exception('Sample size is smaller than the numer of regressors: %s < %s'%(Phi.shape[0], Phi.shape[1]))
            
            unnormalized_infA = np.matmul(Phi.T, Phi)
            if np.linalg.det(unnormalized_infA) == 0:
              print('unnormalized_infA', unnormalized_infA.shape, '\n', unnormalized_infA)
              raise Exception('Degenerate information matrix: det(infA) = 0')
            
            infA =  unnormalized_infA / sample_size
            invertedA = np.linalg.inv(infA + ls_lambda * np.eye(Phi.shape[1]))
            coeffs = np.matmul(invertedA, np.matmul(Phi.T, trainY) / sample_size)

            if verbose >= 2:
                print('infA:\n', infA)
                print('\ninvertedA:\n', invertedA)

            
        elif method == 'projection':
            coeffs = np.matmul(Phi.T/sample_size, trainY) #!!!!???

            if verbose >= 2:
                print('Projection calculation:\n', Phi.T, trainY)

        else:
            raise Exception('Unknown training method!')

        self.set_coef_(coeffs)

 
    
    def info(self, extended=True):
        pce_info = 'PCE model info: \n'
        description = self.__str__()
        pce_info += '  %s\n\n'%description
        pce_info += '  input dimension: %s\n'%self.input_dim
        pce_info += '  basis truncation: %s\n'%self.basis_truncation
        pce_info += '  basis functions number: %s\n'%self._basis_functions_number
        pce_info += '  main Sobol indices: %s\n'%self._main_sobol_indices
        pce_info += '  total Sobol indices: %s\n'%self._total_sobol_indices

        if extended:
            multiIndices = self._multiIndices
            if len(multiIndices) > 100:
                pce_info += '  multiIndices: %s\n'%str(multiIndices[:50]) + "...,\n" + str(multiIndices[-50:])
            else:
                pce_info += '  multiIndices: %s\n'%multiIndices
        print(pce_info)

    def __str__(self):
        return 'PCE(marginals=%s, basis_truncation=%s)'%(self.marginals,
            self.basis_truncation)
    

    def _check_train_X_Y(self, X, Y):
        if X.shape[0] != Y.shape[0]:
            raise Exception('X.shape[0] != Y.shape[0]')#!!!
        if X.shape[1] != self.input_dim:
            raise Exception('X.shape[1] != self.input_dim')#!!!
        # if X.shape[0] < self._basis_functions_number:
        #     raise Exception()#!!!

    def get_stats(self):
        stats = {}

        coeffs = self.coef_
        
        I = None
        for i, mI in enumerate(self._multiIndices):
            if mI == [0]*self.input_dim:
                I = i

        stats['expectation'] = coeffs[I][0] if I is not None else 0
        stats['variance'] = sum([coeffs[i]**2 for i in range(len(self._multiIndices)) if i != I])[0]
        return stats

    def _check_is_fitted(self):
        if self.coef_ is None:
            raise Exception('PCE model is not trained')

    def _check_input_array(self, X):
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise Exception('X must be 2D array [n_samples, n_features=%s]'%self.input_dim)


    def predict(self, X):
        """Predict using the PCE model
        X : array-like, shape = (n_samples, n_features) Samples
        predicted : array, shape = (n_samples,1) Returns predicted values
        """
        self._check_is_fitted()
        self._check_input_array(X)
        predicted = self._model(X) 
        return predicted

    def get_sobol_indices(self, ind_type='main', full=False):

        self._check_is_fitted()

        multiIndices = self._multiIndices
        coeffs = self.coef_
        dim = self.input_dim

        var = sum([c[0]**2 for i, c in enumerate(coeffs) if multiIndices[i] != [0]*dim])


        if not full:

            def is_power_multiindex(multiIndex, variable_num):
                if multiIndex[variable_num] > 0:
                    multiIndex_copy = multiIndex.copy()
                    multiIndex_copy[variable_num] = 0
                    if multiIndex_copy == [0]*len(multiIndex_copy):
                        return True
                else:
                    return False
            
            sobolIndices = []
            for variable_num in range(dim):

              if ind_type == 'main':   
                var_j =  sum([c[0]**2 for i, c in enumerate(coeffs) 
                    if is_power_multiindex(multiIndices[i], variable_num)])
                sobolIndices.append(var_j / var)
              
              elif ind_type == 'total':
                var_j =  sum([c[0]**2 for i, c in enumerate(coeffs) 
                    if multiIndices[i][variable_num] > 0])
                sobolIndices.append(var_j / var)

        else:

            sobolIndices = {}
            vars_list = range(1, dim+1)
            var_groups = powerset(vars_list)

            for g in var_groups:

                var_g = 0

                if ind_type == 'main':   

                    for i, c in enumerate(coeffs):

                        if np.prod([multiIndices[i][variable_num-1] > 0 for variable_num in g]) == 1 and\
                               np.sum([multiIndices[i][variable_num-1] > 0 for variable_num in vars_list 
                                                                          if variable_num not in g]) == 0:
                            var_g += c[0]**2
         
              
                elif ind_type == 'total':

                    for i, c in enumerate(coeffs):

                        if sum([multiIndices[i][variable_num-1] > 0 for variable_num in g]) > 0:
                            var_g += c[0]**2

                sobolIndices[g] = var_g / var


        return sobolIndices
    
    def validate(self, testX, testY):

        predictedTestY = self.predict(testX)
        errors = err.get_errors(true = testY, prediction = predictedTestY)
        return errors



def test_pce():

    print('\n================ Test 1 ====================\n')

    marginals = distr.Marginals([distr.Uniform(), distr.Normal()])
    #basis_truncation = {'total_degree':1, 'q_norm':1.0}
    basis_truncation = {'max_degree':1}

    model = PCE(marginals, basis_truncation)
    print(model)
    model.info()
    print('\n')
    print(dir(PCE))
    print('\n')
    #print(model._multiIndices)

    X = np.array([[1,2], [3,7], [9,10], [10, 12], [14, 19]])
    Y = np.array([[1, 3, 2, 5, 7]]).T

    model.fit(X, Y)
    model.fit(X, Y, method='projection')
    predicted_Y = model.predict(X)
    
    model.info()
    print('\n')

    print(model)
    print('predicted_Y =',predicted_Y)

    
    marginals = distr.Marginals([distr.Uniform(a=2, b=10)])
    basis_truncation = {'max_degree':1}

    model = PCE(marginals, basis_truncation)


    X = np.array([[1], [7], [10], [12], [19]])
    Y = np.array([[1, 3, 2, 5, 7]]).T
    model.fit(X, Y)
    model.info()

    err = model.validate(X, Y)
    print(err)

    print('\n============================================\n')



def test_pce_2():

    print('\n================ Test 2 (dim = 1) ====================\n')

    d = 2
    n_test = 10**5
    n = 100000

    left = -1.0 *  math.pi
    right =  math.pi

    def f(X):
        return np.sin(np.sum(X, axis=1))

    ranges = right * np.ones((d, 2))
    ranges[:, 0] = left

    test_X = np.random.rand(n_test, d)
    test_X = (ranges[:, 1] - ranges[:, 0]) * test_X + ranges[:, 0]
    test_Y = f(test_X)#[:, np.newaxis]

    # =======================================================

    marginals = distr.Marginals([distr.Uniform(left, right)]*d)
    #basis_truncation = {'total_degree':1, 'q_norm':1.0}
    basis_truncation = {'max_degree':1}

    model = PCE(marginals, basis_truncation)
    print(model)
    model.info()
    print('\n')
    print(dir(PCE))
    print('\n')

    # =======================================================


    X = (ranges[:, 1] - ranges[:, 0]) * np.random.rand(n, d) + ranges[:, 0]
    Y = f(X)[:, np.newaxis]     

    print(X.shape, Y.shape)

    
    print('*'*60, '\n')
    model.fit(X, Y, method='projection')
    print('Projection c:\n', model.coef_)
    
    err = model.validate(X, Y)
    print('Projection error:', err, '\n') 
    #print('Prjection RRMSE', rrmse_pr)  



    if model._basis_functions_number <= n:
        print('*'*60, '\n')
        model.fit(X, Y, method='ls')
        print('LS c:\n', model.coef_)
        err = model.validate(X, Y)
        print('LS error:', err, '\n') 


    print('\n' + '='*80 + '\n')
    
    print(model.get_sobol_indices(ind_type='main'))
    print(model.get_sobol_indices(ind_type='main', full=True))

    
    print('\n' + '='*80 + '\n')
    print(model.get_sobol_indices(ind_type='total'))
    print(model.get_sobol_indices(ind_type='total', full=True))

    print(model.get_stats())





if __name__ == '__main__':
    # basic
    test_pce()
    
    # dim = 1
    test_pce_2()
    
    # dim = 2
    #test_pce_3()


