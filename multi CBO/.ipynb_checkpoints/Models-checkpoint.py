import numpy as np
import pickle ##w 자료형
import os ##w 내 컴퓨터 디렉토리 등 경로 다뤄줌
from utils import simplex_proj 

import random

import yaml


def norm_Psi(Psi):
    row_sums = Psi.sum(axis=1, keepdims=True)  # Calculate the sum of each row
    Psi = Psi / row_sums  # Normalize each element by dividing by the row sum
    return Psi

def gen_Psi(X,beta,alpha,L1,L2,r = None, normalize = True):
    n = X.shape[0]
    Psi = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Psi[i][j] = alpha[i]*L1(X[j])+(1-alpha[i])*L2(X[j])
#    psi = np.zeros((1,n))
#    for j in range(n):
#        psi[0][j]= L(X[j])
#    Psi = np.tile(psi, (n, 1))
    Psi = Psi - np.min(Psi)
    Psi = -beta* Psi
    Psi = np.exp(Psi)
    if r:
        Psi = step_array(X,r)*Psi
    if normalize == True:
        Psi = norm_Psi(Psi)
    return Psi

def step_array(vectors,r):
    # Convert the list of vectors to a NumPy array
    vectors = np.array(vectors)
    
    # Expand vectors into two 3D arrays for vectorized subtraction
    diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]
    
    # Compute the Euclidean distances
    distances = np.sum(diff**2, axis=-1)
    return distances < r**2




def get_Xmbmp(X, L, beta):
    nump = len(X)
    LL = np.zeros(nump)
    for i in range(nump):
        LL[i] = L(X[i])
    LL = LL - np.min(LL) # exp 폭발 방지 위해 약분
    bmf = np.exp(-beta * LL)  # boltzmann factor
    bmz = np.sum(bmf)  # boltzmann Z
    bmp = bmf / bmz  # boltzmann prob=bmf_i/Z #w 이걸로 내적을 하려고함
    Xm = np.dot(np.transpose(bmp), X) #w 내적한 결과 (X*)
    if np.isnan(Xm).any():  # .any()는 하나라도 참인게 있으면 True
        print(LL)
        print(bmf)
        print(bmz)
        raise ValueError
    return Xm, bmp

def new_get_Xm(X, L, alpha, beta):
    nump = len(X)
    LL = np.zeros((nump,nump))
    for i in range(nump):
        for j in range(nump):
            LL[i][j] = L(X[i],alpha[j])
    LL = LL - np.min(LL) # exp 폭발 방지 위해 약분
    bmf = np.exp(-beta * LL)  # boltzmann factor, NUMP*NUMP 행렬
    bmz = np.sum(bmf, axis = 0)  # boltzmann Z nump짜리 벡터
    bmp = np.divide(bmf,bmz)  # boltzmann prob = bmf의 각 i번째 열을 bmz의 i번째 원소로 나누려고 함 #w 이걸로 내적을 하려고함
    Xm = np.dot(np.transpose(bmp), X) #w 내적한 결과 (X*)
    if np.isnan(Xm).any():  # .any()는 하나라도 참인게 있으면 True
        print(LL)
        print(bmf)
        print(bmz)
        raise ValueError
    return Xm, bmp

class new2_CBO_model():
    # def __init__(self, L, dt=0.1, beta=10.0, lam=0.5, sigma=0.4, lam1=1, proj=True, beta1=0.9, beta2=0.99):
    def __init__(self, L1, L2, config):
        self.L1 = L1
        self.L2 = L2
        self.config = config 
        self.dt = config['dt']
        self.beta = config['beta']
        self.lam = config['lam']
        self.sigma = config['sigma']
        self.lam1 = config['lam1']
        self.proj = config['proj']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.avg = config['avg']
        self.simname = config['simname']
        self.path = "result/" + self.simname + "/"
        self.history = {}
        # self.M=0
        # self.V=0

    def step(self, X, alpha, L1 = None, L2 = None):
        if not L1:
            L1 = self.L1
        if not L2:
            L2 = self.L2
        L1 = self.L1
        L2 = self.L2
        nump=len(X)
        Psi = gen_Psi(X,self.beta,alpha,L1,L2) 
        sqrtdt = np.sqrt(self.dt)
        drift = self.dt*self.lam*np.dot(Psi,X)-self.dt*self.lam*np.dot(np.diag(np.sum(Psi,axis=1)),X)
        diffu = sqrtdt*self.sigma*(np.dot(Psi,X)-np.dot(np.diag(np.sum(Psi,axis=1)),X))*np.random.randn(X.shape[0],X.shape[1])
     
        Xnew = X + drift + diffu
        if self.proj:
            for i in range(nump):
                Xnew[i] = simplex_proj(Xnew[i])
        return Xnew
    
    def weighted_avg(self, X, alpha, L1 = None, L2 = None):
        if not L1:
            L1 = self.L1
        if not L2:
            L2 = self.L2
        L1 = self.L1
        L2 = self.L2
        return np.dot(gen_Psi(X,self.beta,alpha,L1,L2), X)
        
    
    
    def trace_func(self, x, func=lambda x: x, funcname="coord"):
        if funcname not in self.history:
            self.history[funcname] = []
        self.history[funcname].append(func(x))

    def save_func(self, funcname="coord", simnum=None):
        if simnum is None:
            filename = self.path + funcname + ".pkl"
        else:
            filename = self.path + funcname + "_" + str(simnum) + ".pkl"
        with open(filename, 'wb') as file_object:
            pickle.dump(self.history[funcname], file_object)

    def save_config(self):
        file_path = self.path + "config.yaml"
        with open(file_path, 'w') as yaml_file:
            yaml.safe_dump(self.config, yaml_file)

    def make_path(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("The new directory is created!")
        # else:
        # print("directory is already created!")

    def best_loss(self, x, L):
        N = len(x)
        i = 0
        Lx = np.zeros(N)
        while i < N:
            x_i = x[i]
            Lx[i] = L(x_i) #w Lx[i] = L(x[i])로 안하고?
            i += 1
        return np.min(Lx)

class new_CBO_model():
    # def __init__(self, L, dt=0.1, beta=10.0, lam=0.5, sigma=0.4, lam1=1, proj=True, beta1=0.9, beta2=0.99):
    def __init__(self, L, config):
        self.L = L
        self.config = config ##w config?
        self.dt = config['dt']
        self.beta = config['beta']
        self.lam = config['lam']
        self.sigma = config['sigma']
        self.lam1 = config['lam1']
        self.proj = config['proj']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.avg = config['avg']
        self.simname = config['simname']
        self.path = "result/" + self.simname + "/"
        self.history = {}
        # self.M=0
        # self.V=0

    def step(self, X, alpha, L=None):
        if not L:
            L = self.L
        sqrtdt = np.sqrt(self.dt)
        nump=len(X)
        Xm,bmp = new_get_Xm(X, L, alpha, self.beta)# 원래 X는 행렬, Xm이 벡터고, X-Xm에서는 벡터가 자가복제되면서 행렬이 되고, 따라서 연산 가능함
        drift = -self.lam * (X - Xm) * self.dt
        diffu = -self.sigma * (X - Xm) * np.random.randn(X.shape[0], X.shape[1]) * sqrtdt

        Xnew = X + drift + diffu
        if self.proj:
            for i in range(nump):
                Xnew[i] = simplex_proj(Xnew[i])
        return Xnew

    def trace_func(self, x, func=lambda x: x, funcname="coord"):
        if funcname not in self.history:
            self.history[funcname] = []
        self.history[funcname].append(func(x))

    def save_func(self, funcname="coord", simnum=None):
        if simnum is None:
            filename = self.path + funcname + ".pkl"
        else:
            filename = self.path + funcname + "_" + str(simnum) + ".pkl"
        with open(filename, 'wb') as file_object:
            pickle.dump(self.history[funcname], file_object)

    def save_config(self):
        file_path = self.path + "config.yaml"
        with open(file_path, 'w') as yaml_file:
            yaml.safe_dump(self.config, yaml_file)

    def make_path(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("The new directory is created!")
        # else:
        # print("directory is already created!")

    def best_loss(self, x, L):
        N = len(x)
        i = 0
        Lx = np.zeros(N)
        while i < N:
            x_i = x[i]
            Lx[i] = L(x_i) #w Lx[i] = L(x[i])로 안하고?
            i += 1
        return np.min(Lx)

class CBO_model():
    # def __init__(self, L, dt=0.1, beta=10.0, lam=0.5, sigma=0.4, lam1=1, proj=True, beta1=0.9, beta2=0.99):
    def __init__(self, L, config):
        self.L = L
        self.config = config ##w config?
        self.dt = config['dt']
        self.beta = config['beta']
        self.lam = config['lam']
        self.sigma = config['sigma']
        self.lam1 = config['lam1']
        self.proj = config['proj']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.avg = config['avg']
        self.simname = config['simname']
        self.path = "result/" + self.simname + "/"
        self.history = {}
        # self.M=0
        # self.V=0

    def step_weight(self, X, L=None):
        if not L:
            L = self.L
        nump = len(X)
        sqrtdt = np.sqrt(self.dt) #w 루트를 왜 씌우는거지?

        Xm, bmp = get_Xmbmp(X, L, self.beta) #w X*와 weight들을 얻음

        X_avg = X.sum(axis=0) / nump
        drift_avg = -self.lam1 * np.dot(np.reshape(bmp, (-1, 1)), np.reshape((X_avg - Xm), (1, -1))) \
                    * nump * self.dt
        # np.reshape 속 -1은 나머지 정보로부터 결정된다는 뜻.

        drift = -self.lam * (X - Xm) * self.dt

        diffu = -self.sigma * (X - Xm) * np.random.randn(X.shape[0], X.shape[1]) * sqrtdt

        Xnew = X + drift + diffu + self.avg * drift_avg #w Milstein method?
        if self.proj:
            for i in range(nump):
                Xnew[i][:-1] = simplex_proj(Xnew[i][:-1])
        return Xnew

    def step(self, X, L=None):
        if not L:
            L = self.L
        nump = len(X)
        sqrtdt = np.sqrt(self.dt)

        Xm, bmp = get_Xmbmp(X, L, self.beta)

        drift = -self.lam * (X - Xm) * self.dt

        X_avg = X.sum(axis=0) / nump
        Xm, bmp = get_Xmbmp(X, L, self.beta)
        avg_drift = -self.lam1 * self.dt * (X_avg - Xm) * np.ones_like(X)

        diffu = -self.sigma * (X - Xm) * np.random.randn(X.shape[0], X.shape[1]) * sqrtdt

        Xnew = X + drift + self.avg * avg_drift + diffu
        if self.proj:
            for i in range(nump):
                Xnew[i] = simplex_proj(Xnew[i])
        return Xnew

    def trace_func(self, x, func=lambda x: x, funcname="coord"):
        if funcname not in self.history:
            self.history[funcname] = []
        self.history[funcname].append(func(x))

    def save_func(self, funcname="coord", simnum=None):
        if simnum is None:
            filename = self.path + funcname + ".pkl"
        else:
            filename = self.path + funcname + "_" + str(simnum) + ".pkl"
        with open(filename, 'wb') as file_object:
            pickle.dump(self.history[funcname], file_object)

    def save_config(self):
        file_path = self.path + "config.yaml"
        with open(file_path, 'w') as yaml_file:
            yaml.safe_dump(self.config, yaml_file)

    def make_path(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("The new directory is created!")
        # else:
        # print("directory is already created!")

    def best_loss(self, x, L):
        N = len(x)
        i = 0
        Lx = np.zeros(N)
        while i < N:
            x_i = x[i]
            Lx[i] = L(x_i) #w Lx[i] = L(x[i])로 안하고?
            i += 1
        return np.min(Lx)
