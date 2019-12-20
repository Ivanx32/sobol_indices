import numpy as np


def get_errors(true = None, prediction = None):
  
  errors = {}
  mse = float(np.mean((true - prediction)**2, axis = 0))
  errors['mse'] = mse
  errors['rmse'] = np.sqrt(mse)
  errors['rrmse'] = np.sqrt(mse) / float(np.std(true, axis = 0))

  #errors['rrrms'] = np.sqrt(np.mean((true - prediction)**2, axis = 0)) / (np.percentile(true, 95, axis = 0) - np.percentile(true, 5, axis = 0))
  #errors['max'] = np.max(np.abs((true - prediction)), axis = 0)
  #errors['q99'] = np.percentile(np.abs((true - prediction)), 99, axis = 0)
  #errors['q95'] = np.percentile(np.abs((true - prediction)), 95, axis = 0)
  #errors['nrmse'] = np.sqrt(np.mean((true - prediction)**2, axis = 0)) / np.mean(np.abs(true), axis = 0)

  return errors





if __name__ == '__main__':

  np.random.seed(0)

  n = 10**6
    
  a = 10*np.random.randn(n, 1)
  b = a + 2*np.random.randn(n, 1)

  print (a, '\n\n')

  #print np.std(a, axis = 0)

  er = get_errors(a, b)
  print('Errors:', er, '\n')
  
  print('mse (true = 4):', er['mse'])

  print('rmse (true = 2):', er['rmse'])

  print ('rrmse (true = 0.2):', er['rrmse'])

  # print ('max', er['max'])
  # print ('q99', er['q99'])
  # print ('q95', er['q95'])

  # print ('nrmse', er['nrmse'])