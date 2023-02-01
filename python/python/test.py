from memory_profiler import profile
from joblib import Parallel, delayed


@profile()
def my_func(v):
    a = [v] * (10**2)
    b = [2 * v] * (2 * 10**7)
    del b
    return a


if __name__ == "__main__":
    random_evo = Parallel(
        n_jobs=3, verbose=5, backend="loky", pre_dispatch="1.5*n_jobs"
    )(delayed(my_func)(r) for r in [2, 4, 5])
