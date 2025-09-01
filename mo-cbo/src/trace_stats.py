from pathlib import Path
import pickle
import numpy as np
import json

def _load_trace_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        hist = pickle.load(f)
    return hist  # list over iterations

def _find_trace_files(sim_dir: Path, funcname: str):
    # e.g., coord_0.pkl, best_L_3.pkl ...
    return sorted(sim_dir.glob(f"{funcname}_*.pkl"))

def summarize_scalar_trace(run_root, funcname="best_L"):
    """
    스칼라 트레이스(각 it에서 float)를 가정.
    returns dict: {"iters": T, "mean": (T,), "var": (T,), "std": (T,), "n_sims": S}
    """
    run_root = Path(run_root)
    series = []   # list of (T,)
    # print(sorted(run_root.glob("sim*")))
    for sim_dir in sorted(run_root.glob("sim*")):
        files = _find_trace_files(sim_dir, funcname)
        if not files:
            continue
        # 보통 파일은 1개이나, 여러 개면 첫 번째 선택
        hist = _load_trace_pkl(files[0])
        arr = np.asarray(hist, dtype=float)  # shape (T,)
        if arr.ndim != 1:
            raise ValueError(f"{funcname} expected scalar trace per iter, got shape {arr.shape} in {files[0]}")
        series.append(arr)

    if not series:
        raise FileNotFoundError(f"No {funcname}_*.pkl found under {run_root}/sim*/")

    # 길이가 다른 경우를 대비해 최소 T로 맞춤
    # min_T = min(len(a) for a in series)
    # S = len(series)
    # M = np.stack([a[:min_T] for a in series], axis=0)  # (S, T)
    
    # 예시: series = [np.array([1,2,3]), np.array([4,5]), np.array([6])]
    lengths = [len(a) for a in series]
    max_T   = max(lengths)
    avg_T   = np.mean(lengths)
    S = len(series)
    
    M = np.zeros((S, max_T), dtype=series[0].dtype)

    
    for i, a in enumerate(series):
        L = len(a)
        M[i, :L] = a
        if L < max_T:
            M[i, L:] = a[-1]  # 마지막 값으로 채우기
    
#    print(M)

    mean = M.mean(axis=0)          # (T,)
    var  = M.var(axis=0, ddof=0)   # (T,)
    std  = M.std(axis=0, ddof=0)   # (T,)

    return {
        "iters": int(avg_T),
        "n_sims": int(S),
        "mean": mean,
        "var": var,
        "std": std,
    }

def summarize_coord_trace(run_root, funcname="coord", reduce_over="particles_and_sims"):
    """
    coord 트레이스: 각 it에서 (N,D) 배열을 가정.
    reduce_over:
      - "particles_and_sims": 각 it, 각 dim에 대해 (모든 시뮬 + 모든 파티클) 평균/분산
      - "sims_only":          각 it, 각 dim에 대해 (시뮬레이션 평균), 파티클 평균은 각 시뮬 내에서 먼저 함

    returns dict:
      {
        "iters": T,
        "dims": D,
        "mean": (T, D),
        "var":  (T, D),
        "std":  (T, D),
        "n_sims": S,
        "n_particles": N (첫 파일 기준)
      }
    """
    run_root = Path(run_root)
    per_sim = []   # list of (T, N, D)
    N0 = None
    D0 = None
    for sim_dir in sorted(run_root.glob("sim*")):
        files = _find_trace_files(sim_dir, funcname)
        if not files:
            continue
        hist = _load_trace_pkl(files[0])      # list length = T, each (N,D)
        arr = np.asarray(hist)
        if arr.ndim != 3:
            raise ValueError(f"{funcname} expected (T,N,D) trace, got shape {arr.shape} in {files[0]}")
        if D0 is None:
            _, N0, D0 = arr.shape
        per_sim.append(arr)

    if not per_sim:
        raise FileNotFoundError(f"No {funcname}_*.pkl found under {run_root}/sim*/")

    # 길이 다른 경우 최소 T로 맞춤
    min_T = min(a.shape[0] for a in per_sim)
    S = len(per_sim)
    N = per_sim[0].shape[1]
    D = per_sim[0].shape[2]
    # (S, T, N, D)
    M = np.stack([a[:min_T] for a in per_sim], axis=0)

    if reduce_over == "particles_and_sims":
        # 모든 시뮬 + 모든 파티클에 대해 평균/분산
        mean = M.mean(axis=(0, 2))            # (T, D)
        var  = M.var(axis=(0, 2), ddof=0)     # (T, D)
        std  = M.std(axis=(0, 2), ddof=0)     # (T, D)
    elif reduce_over == "sims_only":
        # 먼저 각 시뮬 내에서 파티클 평균: (S,T,N,D) -> (S,T,D)
        M_s = M.mean(axis=2)                  # (S, T, D)
        mean = M_s.mean(axis=0)               # (T, D)
        var  = M_s.var(axis=0, ddof=0)        # (T, D)
        std  = M_s.std(axis=0, ddof=0)        # (T, D)
    else:
        raise ValueError("reduce_over must be 'particles_and_sims' or 'sims_only'")

    return {
        "iters": int(min_T),
        "dims": int(D),
        "n_sims": int(S),
        "n_particles": int(N),
        "mean": mean,
        "var": var,
        "std": std,
    }

def save_stats_json(stats: dict, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # numpy -> list 변환
    def _to_serializable(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    with open(out_path, "w") as f:
        json.dump({k: _to_serializable(v) for k, v in stats.items()}, f, indent=2)
