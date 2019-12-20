import math
import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt


class Distribution():
    
    def __init__(self):
        pass

class Marginals(Distribution):
    def __init__(self, distr_list=[]):
        if not isinstance(distr_list, list):
            raise Exception('Input argument for Marginals should be of list type, proveded %s'%type(distr_list))

        self.distr_list = distr_list
        self.dim = len(distr_list)

    def add(self, distr):
        self.distr_list += [distr]   
        self.dim = len(self.distr_list)

    def get_samples(self, number):
        if len(self.distr_list) == 0:
            raise Exception('Marginals contains empty distributions list')
        samples = np.nan * np.ones((number, self.dim))
        for dim_num, distr in enumerate(self.distr_list):
            samples[:, dim_num] = distr.get_samples(number)
        return samples
    
    def check_multiIndex(self, multiIndex):
        if not isinstance(multiIndex, list):
            raise Exception('multiIndex must be list')
        if len(multiIndex) != self.dim:
            raise Exception('Wrong multiIndex length, should be %s'%self.dim)
        for ind in multiIndex:
            if ind < 0:
                raise Exception('multiIndex contains negative index')

    def get_orthonormal_polynomial(self, multiIndex):
        self.check_multiIndex(multiIndex)
        
        def polynomial_multivariate(x, multiIndex):
            values = np.ones((1, len(x)))
            for dim_num, distr in enumerate(self.distr_list):
                degree = multiIndex[dim_num]
                p = distr.get_orthonormal_polynomial(degree)
                values *= p(x[:, dim_num])
            return values.T

        return lambda x: polynomial_multivariate(x, multiIndex)


    def __str__(self):
        return 'Marginals(['+", ".join([str(d) for d in self.distr_list])+'])'


class Normal(Distribution):
    def __init__(self, mean=0, std=1.0):
        if std <= 0:
            raise Exception('Normal(mean=%s, std=%s): std should be > 0'%(mean, std))
        self.mean = mean
        self.std = std
        self.poly_type = 'Hermite'

    def __str__(self):
        return 'Normal(mean=%s, std=%s, %s)'%(self.mean, self.std, self.poly_type)

    def get_orthonormal_polynomial(self, degree):

        def polynomial_hermitenorm(x, degree):
            x = (x-self.mean)/self.std
            values = scipy.special.eval_hermitenorm(degree, x) 
            values *= 1.0 / math.sqrt(scipy.math.factorial(degree)) #  1.0/math.sqrt(self.std) *        
            return values

        return lambda x: polynomial_hermitenorm(x, degree)


    def get_samples(self, number):
        return np.random.normal(loc=self.mean, scale=self.std, size=number)




class Uniform(Distribution):
    def __init__(self, a=0, b=1.0, poly_type='Legendre'):        
        if b - a < 0:
            raise Exception('Uniform(a=%s, b=%s): b should be > a'%(a, b)) 
        self.a = a
        self.b = b
        self.poly_type = poly_type


    def __str__(self):
        return 'Uniform(a=%s, b=%s, %s)'%(self.a, self.b, self.poly_type)

    def get_orthonormal_polynomial(self, degree):

        if degree < 0:
            raise Exception('The degree should be >= 0') 
        
        def polynomial_legendre(x, degree):
            x = (2.0*x - (self.a+self.b))/(float(self.b) - self.a)
            values = scipy.special.eval_legendre(degree, x) # * math.sqrt(2.0 / (self.b - self.a)) 
            values *= math.sqrt(2.0*degree+1.0)
            return values

        def polynomial_trigonometric(x, degree):
            
            x = (x - self.a)/(float(self.b) - self.a)
            
            if degree == 0:
                values = np.ones(len(x))
            elif degree % 2 == 1:
                values = math.sqrt(2) * np.sin((degree+1) * math.pi * x)
            else:
                values = math.sqrt(2) * np.cos(degree * math.pi * x)
    
            return values



        if self.poly_type == 'Legendre':
            poly =  lambda x: polynomial_legendre(x, degree)
        
        elif self.poly_type == 'Trigonometric':
            poly =  lambda x: polynomial_trigonometric(x, degree)
            #print('Trig') #!!!
        
        else:
            raise Exception('Unknown polynomial type for Uniform distribution.')
        
        return poly

        
    def get_samples(self, number):
        return np.random.uniform(low=self.a, high=self.b, size=number)
      


def inner_product(distribution, p1,p2, points_number=10**6):
    samples = distribution.get_samples(points_number)
    product = np.mean(p1(samples) * p2(samples))
    return product


def test_orthonormality(distribution, max_degree=3, thr=10**-2):
    res = np.nan * np.ones((max_degree+1, max_degree+1))
    for i in range(max_degree+1):
        for j in range(max_degree+1):
            p_i = distribution.get_orthonormal_polynomial(i)
            p_j = distribution.get_orthonormal_polynomial(j)
            res[i, j] = inner_product(distribution, p_i,p_j)
    
    # % np.linalg.norm(res-np.eye(max_degree+1))
    def norm(A):
        return np.max(np.abs(A))

    if norm(res-np.eye(max_degree+1)) < thr:
        print('Orthonormality test is PASSED')
    else:
        print('Orthonormality test is FAILED')
    return res




if __name__ == '__main__':

    d7 = Uniform(-1, 3, poly_type='Trigonometric')

    d8 = Uniform(-1, 3)

    # m = Marginals([d7,d8])
    # res = test_orthonormality(m, max_degree=3, thr=10**-2)
    # print(res, '\n')

    
    print(d7, d8)
    res = test_orthonormality(d7, max_degree=3, thr=10**-2)
    print(res, '\n')

    res = test_orthonormality(d8, max_degree=3, thr=10**-2)
    print(res)
    

    zxc
    
    d = Normal(mean=10)
    d1 = Normal(mean=10)
    d2 = Uniform(0, 2)

    n  = 100

    m = Marginals([d1,d2])
    print(d, d1,d2)
    p = d.get_orthonormal_polynomial(degree=3)
    a = np.array(np.random.rand(10000,1))
    print(p(a))
    print(a.shape)
    
    print(m)
    d = Uniform(-1, 1)
    p0 = lambda x: scipy.special.eval_legendre(0, x)
    p1 = lambda x: scipy.special.eval_legendre(1, x)
    p2 = lambda x: scipy.special.eval_legendre(2, x)
    inner = inner_product(d, p0, p0)
    print('inner: ', inner)


    p = d.get_orthonormal_polynomial(degree=1)

    X = np.linspace(-1,1,100)
    plt.plot(X, p(X))
    

    inner = inner_product(d, p1, p1)
    print('inner: ', inner)

    inner = inner_product(d, p1, p2)
    print('inner: ', inner)
    

    print(Normal().get_samples(10))
    print(Uniform().get_samples(10))

    print(Marginals([Normal(mean=-1, std=1.0), Uniform(a=0, b=2)]).get_samples(10))
    Marginals([Normal(mean=-1, std=1.0), Uniform(a=0, b=2)]).get_samples(10**8)
    
    m = Marginals()
    for _ in range(10):
        m.add(Normal())

    print(m)
    s = m.get_samples(15)
    print(s.shape)

    print(Marginals().get_samples(1))
    d = Normal(mean=3, std=7)
    d = Uniform(a = -5, b= 11)
    
    s = d.get_samples(n)

    p1 = d.get_orthonormal_polynomial(1)
    p2 = d.get_orthonormal_polynomial(2)
    p3 = d.get_orthonormal_polynomial(3)

    #print(np.mean(p1(s)*p1(s)))

    d = Uniform(a =-5, b= 11)
    d = Uniform(a=0, b=1)
    d = Uniform()
    d = Uniform(a =-500, b= 1100)

    d = Normal(mean=300, std=70)
    d = Normal()

    res = test_orthonormality(d)
    res = test_orthonormality(m)
    print(res)
    m = Marginals([Normal()])

    p1 = m.get_orthonormal_polynomial(multiIndex=[1])
    p2 = m.get_orthonormal_polynomial(multiIndex=[2])

    res = inner_product(m, p1,p1)
    print(res)

    res = inner_product(m, p1,p2)
    print(res)


    plt.show()













