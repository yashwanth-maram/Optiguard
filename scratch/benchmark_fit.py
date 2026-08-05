import numpy as np
import time

def benchmark():
    N = 4096
    max_iter = 20
    J = np.random.randn(N, 128, 4).astype(np.float32)
    res = np.random.randn(N, 128).astype(np.float32)
    
    # 1. Matmul
    t0 = time.time()
    for _ in range(max_iter):
        JT = np.swapaxes(J, 1, 2)
        JTJ = np.matmul(JT, J)
        JTr = np.matmul(JT, res[:, :, None])[:, :, 0]
    t1 = time.time()
    
    # 2. Unrolled
    t2 = time.time()
    for _ in range(max_iter):
        JTJ = np.empty((N, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(i, 4):
                val = np.sum(J[..., i] * J[..., j], axis=1)
                JTJ[:, i, j] = val
                if i != j:
                    JTJ[:, j, i] = val
                    
        JTr = np.empty((N, 4), dtype=np.float32)
        for i in range(4):
            JTr[:, i] = np.sum(J[..., i] * res, axis=1)
    t3 = time.time()
    
    print(f"Matmul: {(t1 - t0)*1000:.2f} ms")
    print(f"Unrolled: {(t3 - t2)*1000:.2f} ms")

if __name__ == "__main__":
    benchmark()
