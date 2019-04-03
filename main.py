import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from importance_sampling.core import ImportanceSampling
from importance_sampling.helpers import create_and_save_plot
import pandas as pd


def define_run():
    quantile = 99.95
    sample_sizes = [5000,
                    10000,
                    50000,
                    100000,
                    500000,
                    1000000]
    shifts = np.linspace(0, 6, 13)
    shifts = np.sort(np.append(shifts, norm.ppf(quantile / 100.0)))
    sim_sizes = 1000
    kwargs = dict(quantile=quantile,
                  sample_sizes=sample_sizes,
                  shifts=shifts,
                  sim_sizes=sim_sizes,
                  pool_size=10)
    return kwargs


if __name__ == '__main__':
    preload = True
    my_kwargs = define_run()
    my_df = pd.read_excel('results.xlsx') if preload else ImportanceSampling(**my_kwargs).run()
    my_fig = create_and_save_plot(args=my_kwargs, df=my_df, figsize=12, dpi=500)
    plt.show(my_fig)
