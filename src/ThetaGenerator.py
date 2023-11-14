import numpy as np

class ThetaGenerator():
    mu = []
    var = []
    def __init__(self, dim, noise_var):
        self.dim=dim
        self.noise_var=noise_var

    #事後平均、事後分散の計算
    def calc(self, phi, y):
        A = np.dot(phi.T, phi) + self.noise_var * np.eye(self.dim)
        Ainv=np.linalg.inv(A)
        Ainv_phi_T = np.dot(Ainv, phi.T)
        self.mu = np.dot(Ainv_phi_T, y)
        self.var = self.noise_var * Ainv
    
    def getparam(self):
        return self.mu, self.var
    
    def calc_init(self, phi):
        self.mu = np.zeros(phi.shape[1]).reshape(phi.shape[1],1)
        self.var = np.eye(phi.shape[1])

    def getTheta(self, num):
        L = np.linalg.cholesky(self.var)
        z = np.random.randn(self.var.shape[1],num)
        tmp = np.dot(L, z)
        theta = self.mu.ravel() + tmp.T
        return theta