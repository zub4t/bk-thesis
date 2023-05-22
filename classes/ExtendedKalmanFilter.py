import numpy as np
import KalmanFilter
class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        super().__init__(F=F, B=B, H=H, Q=Q, R=R, P=P, x0=x0)

    def update(self, z, HJacobian, Hx, args=(), hx_args=()):
        y = z - Hx(*hx_args)
        S = self.R + np.dot(HJacobian(self.x, *args), np.dot(self.P, HJacobian(self.x, *args).T))
        K = np.dot(np.dot(self.P, HJacobian(self.x, *args).T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, HJacobian(self.x, *args)), self.P), (I - np.dot(K, HJacobian(self.x, *args)).T)) + np.dot(np.dot(K, self.R), K.T)

