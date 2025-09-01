import numpy as np
import pickle ##w 자료형
import os ##w 내 컴퓨터 디렉토리 등 경로 다뤄줌
from src.utils import simplex_proj 

import random

import yaml

import numpy as np

# ---------- Psi 정규화 ----------
def norm_Psi(Psi, eps=1e-12):
    row_sums = Psi.sum(axis=1, keepdims=True)
    # 행합이 0인 줄은 균등분포로 대체 (혹은 그대로 0 유지 후 나중 처리도 가능)
    zero_rows = (row_sums <= eps)
    Psi_norm = np.empty_like(Psi)
    # 기본: 안전 나눗셈
    Psi_norm = Psi / np.where(row_sums > eps, row_sums, 1.0)
    # 완전 0행은 균등 할당
    if zero_rows.any():
        n = Psi.shape[1]
        Psi_norm[zero_rows[:, 0]] = 1.0 / n
    return Psi_norm

# ---------- 반경 마스크 ----------
def step_array(vectors, r, include_self=False):
    vectors = np.asarray(vectors)               # (n, d)
    diff = vectors[:, None, :] - vectors[None, :, :]
    distances2 = np.sum(diff * diff, axis=-1)   # (n, n)
    mask = distances2 < (r * r)
    if not include_self:
        np.fill_diagonal(mask, False)           # 자기연결 제외 (원하면 True)
    return mask.astype(float)                   # 0/1 행렬로 사용

# ---------- Psi 생성 (수치안정 + 마스킹 + 정규화) ----------
def gen_Psi(X, beta, alpha, L1, L2, r=None, normalize=True, use_logsumexp=True):
    """
    X: (n, d)
    alpha: (n,) 또는 (n,1) - i행에서 쓰일 alpha[i] (0~1 가정)
    L1,L2: 콜러블
    r: None 또는 반경(float)
    """
    X = np.asarray(X)
    n = X.shape[0]
    alpha = np.asarray(alpha).reshape(-1)

    # 비용행렬 Psi_raw[i,j] = alpha[i]*L1(X[j]) + (1-alpha[i])*L2(X[j])
    # j에 대해서 L1/L2는 한 번만 계산해 캐시
    L1_vals = np.array([L1(xj) for xj in X])    # (n,)
    L2_vals = np.array([L2(xj) for xj in X])    # (n,)
    base = np.vstack((L1_vals, L2_vals)).T      # (n, 2) for j

    Psi_raw = np.empty((n, n), dtype=float)
    # 행별 alpha[i]로 선형결합
    # Psi_raw[i, j] = a_i * L1_vals[j] + (1-a_i) * L2_vals[j]
    Psi_raw = alpha[:, None] * base[:, 0][None, :] + (1.0 - alpha)[:, None] * base[:, 1][None, :]

    # 수치 안정화: 볼츠만 가중치
    # exp(-beta * (Psi_raw - row_min))
    row_min = Psi_raw.min(axis=1, keepdims=True)
    logits = -beta * (Psi_raw - row_min)        # (n, n), 최대가 0
    # 반경 마스크 적용(있으면)
    if r is not None:
        M = step_array(X, r, include_self=False)  # (n, n) 0/1
        logits = logits + np.log(M + 1e-30)       # 마스크 0인 곳은 -inf 효과

    if normalize:
        if use_logsumexp:
            # row-wise log-sum-exp
            max_logit = np.max(logits, axis=1, keepdims=True)  # <= 0
            stable = logits - max_logit
            np.exp(stable, out=stable)
            denom = np.sum(stable, axis=1, keepdims=True)
            # denom==0 처리(마스크가 모두 0인 행 등): 균등분포
            bad = denom <= 0.0
            Psi = np.divide(stable, denom, where=~bad, out=np.zeros_like(stable))
            if bad.any():
                Psi[bad[:, 0]] = 1.0 / n
        else:
            Psi = np.exp(logits)
            Psi = norm_Psi(Psi)
    else:
        Psi = np.exp(logits)  # 비정규화로 사용(라플라시안 등)

    return Psi  # (n, n)

# ---------- 전역 컨센서스용 볼츠만 평균 ----------
def get_Xmbmp(X, L, beta):
    """
    전통 CBO: LL_i = L(X[i]), p_i ∝ exp(-beta * (LL_i - min(LL)))
    return Xm (d,), bmp (n,)
    """
    X = np.asarray(X)
    n, d = X.shape
    LL = np.array([L(xi) for xi in X])               # (n,)
    LL = LL - LL.min()                               # 안정화
    bmf = np.exp(-beta * LL)                         # (n,)
    Z = bmf.sum()
    if not np.isfinite(Z) or Z <= 0:
        # 모두 0이거나 NaN인 경우 대응
        bmp = np.full(n, 1.0 / n)
    else:
        bmp = bmf / Z

    Xm = bmp @ X                                     # (d,)
    if np.isnan(Xm).any():
        print("LL:", LL)
        print("bmf:", bmf)
        print("Z:", Z)
        raise ValueError("NaN in Xm")
    return Xm, bmp

# ---------- new_get_Xm: 전역 컨센서스(옵션 A) ----------
def new_get_Xm(X, L, alpha, beta, alpha_as_weights=True):
    """
    전역 컨센서스 버전.
    아이디어: 'alpha'의 분포를 사용해 각 X[i]의 비용을 통합한 뒤(스칼라),
              볼츠만 가중치를 만들어 Xm을 (d,)로 반환.

    alpha_as_weights=True:
        C_i = sum_j w_j * L(X[i], alpha_j),  w = softmax(alpha) 또는 normalize(alpha)
    alpha_as_weights=False:
        C_i = L(X[i], alpha_i)  (대각 인덱스 일치)

    반환: Xm (d,), bmp (n,)
    """
    X = np.asarray(X)
    n, d = X.shape
    alpha = np.asarray(alpha).reshape(-1)

    if alpha_as_weights:
        # alpha를 확률로 정규화(합=1). 음수가 가능하면 softmax로 바꾸세요.
        wsum = alpha.sum()
        if wsum <= 0:
            w = np.full(n, 1.0 / n)
        else:
            w = alpha / wsum
        # C_i = sum_j w_j * L(X[i], alpha_j)
        C = np.zeros(n)
        for i in range(n):
            # 필요시 벡터화 가능
            C[i] = sum(w[j] * L(X[i], alpha[j]) for j in range(n))
    else:
        # 대각만 사용: C_i = L(X[i], alpha[i])
        C = np.array([L(X[i], alpha[i]) for i in range(n)])

    # 볼츠만
    C = C - C.min()
    bmf = np.exp(-beta * C)           # (n,)
    Z = bmf.sum()
    bmp = bmf / Z if Z > 0 else np.full(n, 1.0 / n)

    Xm = bmp @ X                      # (d,)
    if np.isnan(Xm).any():
        print("C:", C)
        print("bmf:", bmf)
        print("Z:", Z)
        raise ValueError("NaN in Xm")
    return Xm, bmp


import os, pickle, yaml
import numpy as np

class _BaseCBO:
    def __init__(self, config):
        self.config = config
        self.dt     = config['dt']
        self.beta   = config['beta']
        self.lam    = config['lam']
        self.sigma  = config['sigma']
        self.lam1   = config.get('lam1', 0.0)
        self.proj   = config.get('proj', False)
        self.beta1  = config.get('beta1', 0.9)   # (미사용이면 제거 고려)
        self.beta2  = config.get('beta2', 0.99)  # (미사용이면 제거 고려)
        self.avg    = config.get('avg', 0.0)
        self.simname = config['simname']
        self.noise_type = config.get("noise_type", "homo")  # "homo" or "hetero"
        self.path    = os.path.join("results", self.simname)
        self.history = {}

        # 디렉토리 보장
        os.makedirs(self.path, exist_ok=True)

        # 재현성 옵션 (있으면 사용)
        seed = config.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

    def trace_func(self, x, func=lambda x: x, funcname="coord"):
        self.history.setdefault(funcname, []).append(func(x))

    def save_func(self, funcname="coord", simnum=None):
        fname = f"{funcname}.pkl" if simnum is None else f"{funcname}_{simnum}.pkl"
        with open(os.path.join(self.path, fname), 'wb') as f:
            pickle.dump(self.history.get(funcname, []), f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_config(self):
        with open(os.path.join(self.path, "config.yaml"), 'w') as f:
            yaml.safe_dump(self.config, f)
            
    def make_path(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("The new directory is created!")

    @staticmethod
    def _project_simplex_rows(X, b=1.0):              
        """
        각 행을 유클리드 심플렉스 { y >= 0, sum(y)=b }에 사영.
        Wang & Carreira-Perpiñán(2013) 알고리즘을 행 단위로 벡터화.
    
        Args:
            X (array_like): (N, d) 입력 행렬
            b (float): 심플렉스 반지름(기본 1.0, 양수)
        Returns:
            np.ndarray: (N, d) 사영 결과
        """
        X = np.asarray(X, dtype=float)
        if b <= 0:
            raise ValueError("b must be positive")
        N, d = X.shape
        if d == 0 or N == 0:
            return X.copy()
    
        # 1) 각 행을 내림차순 정렬
        U = np.sort(X, axis=1)[:, ::-1]         # (N, d)
        CSSV = np.cumsum(U, axis=1)             # (N, d)
        j = np.arange(1, d + 1)[None, :]        # (1, d)
    
        # 2) cond: u_j - (cssv_j - b)/(j) > 0  (여기서 j는 1..d)
        cond = U - (CSSV - b) / j > 0           # (N, d), True..True, False..False의 단조 패턴
        # 3) rho = 마지막으로 True인 위치
        #    cond가 단조이므로 True의 개수 - 1 이 마지막 True의 인덱스
        rho = cond.sum(axis=1) - 1              # (N,)
    
        # 4) theta = (sum_{i<=rho} u_i - b) / (rho + 1)
        rows = np.arange(N)
        theta = (CSSV[rows, rho] - b) / (rho + 1)  # (N,)
    
        # 5) y = max(x - theta, 0)
        Y = np.maximum(X - theta[:, None], 0.0)
        return Y


    @staticmethod
    def best_loss(x, L):
        # 벡터화 평가(가능하면)
        vals = np.array([L(xi) for xi in x])
        return vals.min()
        
class Multi_CBO_model(_BaseCBO):
    def __init__(self, L1, L2, config):
        super().__init__(config)
        self.L1 = L1
        self.L2 = L2

    def step(self, X, alpha, L1=None, L2=None):
        # 인자 우선, 없으면 self 사용
        L1 = self.L1 if L1 is None else L1
        L2 = self.L2 if L2 is None else L2

        Psi = gen_Psi(X, self.beta, alpha, L1, L2)  # (N,N) 가중치 행렬 가정
        N, d = X.shape
        sqrtdt = np.sqrt(self.dt)

        # diag(sum(Psi,1)) * X = (row_sum[:,None] * X)로 벡터화
        row_sum = Psi.sum(axis=1, keepdims=True)
        lapX = Psi @ X - row_sum * X  # (N,d), 그래프 라플라시안 형태

        drift = self.lam * lapX * self.dt
        diffu = self.sigma * lapX * (np.random.randn(N, d)) * sqrtdt

        Xnew = X + drift + diffu

        if self.proj:
            Xnew = self._project_simplex_rows(Xnew)
        return Xnew

    def weighted_avg(self, X, alpha, L1=None, L2=None):
        L1 = self.L1 if L1 is None else L1
        L2 = self.L2 if L2 is None else L2
        Psi = gen_Psi(X, self.beta, alpha, L1, L2)   # (N,N)
        return Psi @ X

class new_CBO_model(_BaseCBO):
    def __init__(self, L, config):
        super().__init__(config)
        self.L = L

    def step(self, X, alpha, L=None):
        L = self.L if L is None else L
        N, d = X.shape
        sqrtdt = np.sqrt(self.dt)

        Xm, bmp = new_get_Xm(X, L, alpha, self.beta)  # Xm: (d,), bmp: (N,)
        # Drift: -lambda (X - Xm)
        center = (X - Xm)               # (N,d)
        drift  = -self.lam  * center * self.dt
        diffu  =  self.sigma * center * np.random.randn(N, d) * sqrtdt  # 부호 통일

        Xnew = X + drift + diffu
        if self.proj:
            Xnew = self._project_simplex_rows(Xnew)
        return Xnew

class CBO_model(_BaseCBO):
    def __init__(self, L, config):
        super().__init__(config)
        self.L = L

    def step_weight(self, X, L=None):
        L = self.L if L is None else L
        N, d = X.shape
        sqrtdt = np.sqrt(self.dt)

        Xm, bmp = get_Xmbmp(X, L, self.beta)  # Xm: (d,), bmp: (N,)
        # 가정: bmp.sum() == 1  (볼츠만 가중치 정규화)
        # 그렇지 않다면 아래 스케일링을 재검토해야 합니다.
        if not np.isclose(bmp.sum(), 1.0):
            bmp = bmp / (bmp.sum() + 1e-12)

        center = X - Xm
        drift  = -self.lam * center * self.dt
        # ---- noise 생성 (동질/이질 선택) ----
        if self.noise_type == "homo":
            # 한 타임스텝에 d-차원 잡음 벡터 하나 생성 → 모든 파티클에 공유
            eps = np.random.randn(1, d)       # (1, d), broadcasting → (N, d)
        else:
            # 파티클별 독립 잡음
            eps = np.random.randn(N, d)       # (N, d)

        # sigma는 scalar 또는 (d,) 모두 호환되게 브로드캐스트
        sigma = np.asarray(self.sigma)

        # multiplicative diffusion (기존 정책 유지: center * noise)
        diffu = sigma * center * eps * sqrtdt # (N, d)

        
        # diffu  =  self.sigma * center * np.random.randn(N, d) * sqrtdt

        # "평균에 대한 추가 드리프트" (문헌마다 해석 다름)
        X_avg = X.mean(axis=0)          # (d,)
        avg_drift_vec = -self.lam1 * (X_avg - Xm) * self.dt  # (d,)
        # 개체별 가중 적용: outer(bmp, avg_drift_vec)
        drift_avg = np.outer(bmp, avg_drift_vec) * (N if self.config.get("scale_avg_by_N", False) else 1)

        Xnew = X + drift + diffu + self.avg * drift_avg
        if self.proj:
            Xnew = self._project_simplex_rows(Xnew)
        return Xnew

    def step(self, X, L=None):
        L = self.L if L is None else L
        N, d = X.shape
        sqrtdt = np.sqrt(self.dt)

        Xm, bmp = get_Xmbmp(X, L, self.beta)  # 한번만
        center = X - Xm
        drift  = -self.lam * center * self.dt
        diffu  =  self.sigma * center * np.random.randn(N, d) * sqrtdt

        # 평균 드리프트 (Milstein 아님: Euler–Maruyama + 보조 항)
        X_avg = X.mean(axis=0)          # (d,)
        avg_drift = -self.lam1 * (X_avg - Xm) * self.dt
        avg_drift = np.broadcast_to(avg_drift, X.shape)

        Xnew = X + drift + self.avg * avg_drift + diffu
        if self.proj:
            Xnew = self._project_simplex_rows(Xnew)
        return Xnew
