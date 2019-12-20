import numpy as np
import math
# import orto_basis as orto

import pickle 
import shelve


import pce


# Ishigami

# noise_level = 0.0#0.2#0.05#0.1#1.4#1.0#1.4#0.7#0.0
# Y += noise_level * np.random.normal(size=Y.shape)


# X = (ranges[:, 1] - ranges[:, 0]) * X + ranges[:, 0]???


def ishigami(X, A = 7, B = 0.1):
    # First-order indices: x1: 0.3139, x2: 0.4424, x3: 0.0

    """
         ranges =  np.array([
                        [-math.pi, math.pi],
                        [-math.pi, math.pi],
                        [-math.pi, math.pi]
    ])   
    """
  
    return np.sin(X[:, 0]) + A * np.power(np.sin(X[:, 1]), 2) + \
      B * np.power(X[:, 2], 4) * np.sin(X[:, 0])


def get_indices_ishigami(A = 7, B = 0.1, full=False):


    v = A**2 / 8.0  +  0.5 * (1+B*math.pi**4/5.0)**2 +  B**2 * math.pi**8 * 8.0/225

    if not full:
      
      true_SI = np.array([0.5 * (1+B*math.pi**4/5.0)**2/v, A**2/8.0/v, 0])

      true_TI = np.array([0.5 * (1+B*math.pi**4/5.0)**2/v +  B**2 * math.pi**8 * 8.0/225/v,\
        A**2/8.0/v, B**2 * math.pi**8 * 8.0/225/v])

      result = {'main':true_SI, 'total':true_TI }

    else:

      main_full = {
                    (1,): 0.5 * (1+B*math.pi**4/5.0)**2 / v,
                    (2,): A**2/8.0 / v,
                    (1, 2): 0.0,
                    (3,): 0.0,
                    (1, 3): B**2 * math.pi**8 * 8.0/225 / v,
                    (2, 3): 0.0,
                    (1, 2, 3): 0.0,
                  } 


      total_full = {
                    (1,): 0.5 * (1+B*math.pi**4/5.0)**2/v +  B**2 * math.pi**8 * 8.0/225/v,
                    (2,): A**2/8.0 / v,
                    (1, 2): 1,
                    (3,): B**2 * math.pi**8 * 8.0/225/v,
                    (1, 3): 1 - A**2/8.0 / v,
                    (2, 3): 1 - 0.5 * (1+B*math.pi**4/5.0)**2 / v,
                    (1, 2, 3): 1,
                   } 


      result = {'main': main_full, 'total': total_full}




    return result


def sobol_g_func(a, X):
    """
    ranges = np.array([[-1.0, 1.0],
                        [-1.0, 1.0],
                        [-1.0, 1.0]])
    """
    result = 1
    for i in range(X.shape[1]):
        result *= ( (abs(2 * X[:, i]) + a[i]) / (1 + a[i]))
    return result




def get_indices_sobol_g_func(a, full=False):

  dim = len(a)
  
    
  var_y = 1.0
  for coef in a:
    var_y *= ((3*coef**2 + 6*coef + 4.0) / (3*(coef+1)**2))
  var_y -= 1.0

  if not full:

    result = {'main':[], 'total':[]}

    for ind in range(dim):
      var_x = 1.0 / (3 * (a[ind] + 1.0)**2)

      result['main'].append(var_x / var_y)

      var_x = 1.0
      for i in range(dim):
        if i != ind:
          var_x *= ((3*a[i]**2 + 6*a[i] + 4.0) / (3*(a[i] + 1.0)**2))
      var_x -= 1.0
          
      result['total'].append(1 - var_x / var_y)
    
    result['total'] = np.array(result['total'])
    result['main'] = np.array(result['main'])

  else:

    result = {'main':{}, 'total':{}}

    vars_list = range(1, dim+1)
    var_groups = pce.powerset(vars_list)

    for g in var_groups:

        var = 1
        for variable in g:
            var *= 1.0 / (3 * (a[variable-1] + 1.0)**2)
        result['main'][g] = var / var_y


    def common_elements(a,b):
        c = set(a).intersection(b)
        return bool(c)


    for g in var_groups:
            
        #print([result['main'][gr]/ var_y for gr in var_groups if common_elements(g, gr) ], '\n')
        tot_var = sum([result['main'][gr] for gr in var_groups if common_elements(g, gr) ])
        result['total'][g] = tot_var


  return result
  


if __name__ == '__main__':


  # X = np.random.rand(1000000, 3)

  
  a = [1,2]
  #a  = range(1, 11)


  print(get_indices_sobol_g_func(a, full=True)['main'])
  print('\n')
  print(get_indices_sobol_g_func(a)['main'])
   
  print('\n\n')
  print(get_indices_sobol_g_func(a, full=True)['total'])
  print('\n')
  print(get_indices_sobol_g_func(a)['total'])



  print('\n\n',get_indices_ishigami())
