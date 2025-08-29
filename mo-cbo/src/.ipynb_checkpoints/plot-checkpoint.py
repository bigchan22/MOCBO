# src/plot.py
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_coord(coord_pkl_path):
    """
    coord 피클을 (T, N, D) ndarray로 로드.
    save_func('coord', simnum=K)로 저장된 파일을 기대.
    """
    coord_pkl_path = Path(coord_pkl_path)
    with open(coord_pkl_path, "rb") as f:
        hist = pickle.load(f)     # list of (N,D)
    coord = np.asarray(hist)      # (T, N, D)
    return coord

def find_first_coord(sim_dir, pattern="coord_*.pkl"):
    """
    sim0000 같은 폴더에서 coord 피클 하나를 찾아 경로 반환.
    여러 개면 첫 번째 선택.
    """
    sim_dir = Path(sim_dir)
    files = sorted(sim_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No coord pkl matched '{pattern}' in {sim_dir}")
    return files[0]

def plot_particles_at_iters(coord, iters, out_dir, title_prefix="iter",
                            minimizer=None, equal_aspect=True, dpi=150,
                            xlimylim=None):
    """
    특정 iters에서 입자 산점도를 파일로 저장.
    
    Parameters
    ----------
    coord : np.ndarray (T, N, D)
        저장된 궤적 (iterations × particles × dimension)
    iters : list[int]
        저장할 iteration 인덱스
    out_dir : str or Path
        출력 폴더
    title_prefix : str
        그림 제목 앞에 붙을 문자열
    minimizer : np.ndarray or None
        (N,2) minimizer 좌표 → 검은색 X로 표시
    equal_aspect : bool
        True면 가로/세로 동일 스케일
    dpi : int
        저장 이미지 해상도
    xlimylim : tuple or None
        (xmin, xmax, ymin, ymax) 직접 지정. None이면 coord 전체 범위 자동 계산.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T, N, D = coord.shape
    if D != 2:
        raise ValueError("현재 함수는 D=2만 지원합니다. (D>2는 PCA 등 차원축소를 쓰세요)")

    if xlimylim is None:
        xmin, xmax = coord[..., 0].min(), coord[..., 0].max()
        ymin, ymax = coord[..., 1].min(), coord[..., 1].max()
    else:
        xmin, xmax, ymin, ymax = xlimylim

    for it in iters:
        if it < 0 or it >= T:
            print(f"[warn] skip it={it} (0..{T-1})")
            continue
        X = coord[it]  # (N,2)
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.8, label='particles')
        if minimizer is not None:
            plt.scatter(minimizer[:,0], minimizer[:,1], c='black', marker='x', s=35, label='minimizer')
            plt.legend()
        plt.title(f"{title_prefix} {it}")
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
        if equal_aspect:
            plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(ls="--", alpha=0.3)
        plt.savefig(out_dir / f"particles_iter{it:04d}.png", dpi=dpi, bbox_inches="tight")
        plt.close()

def plot_every_k(coord, k, out_dir, **kwargs):
    """
    0, k, 2k, ... 프레임을 저장.
    """
    T = coord.shape[0]
    iters = list(range(0, T, k))
    plot_particles_at_iters(coord, iters, out_dir, **kwargs)

def frames_to_gif(frames_dir, gif_path, pattern="particles_iter*.png", fps=6):
    """
    저장된 프레임들을 GIF로 합치기.
    """
    import imageio
    frames_dir = Path(frames_dir)
    files = sorted(frames_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No frames matched '{pattern}' in {frames_dir}")
    images = [imageio.v2.imread(f) for f in files]
    imageio.mimsave(gif_path, images, fps=fps)
