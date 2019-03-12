import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from src.importance_sampling import ImportanceSampling
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
    args = dict(quantile=quantile,
                sample_sizes=sample_sizes,
                shifts=shifts,
                sim_sizes=sim_sizes,
                pool_size=10)
    return args


def create_plot(figsize=7, args=None, df=None):

    if args is None and df is None:
        args = define_run()
        df = pd.read_excel('results.xlsx')

    # Create a pivot table for plotting purposes
    aggfunc = {'std_true': np.sum}
    index = 'shift'          # x-axis
    columns = 'sample_size'  # legend
    values = aggfunc.keys()  # y-axis
    df_pivot = df.pivot_table(values=values, index=index, columns=columns, aggfunc=aggfunc)

    # Create a plot which displays the precision introduced by IS
    fig = plt.figure(figsize=(figsize, int(1080 / 1920 * figsize)))
    ax = fig.gca()
    ax.plot(df_pivot, '-o')
    ax.set_xlabel('Mean shift')
    ax.set_ylabel(f'Standard deviation (based on {args["sim_sizes"]} samples)')
    ax.set_title('Importance Sampling for various mean shifts and sample sizes')
    ax.set_ylim([0, 0.02])
    ax.grid(b=True)
    ax.legend(labels=args["sample_sizes"], title='Sample size')
    fig.savefig('results.png')
    return df, fig


if __name__ == '__main__':
    # my_df = ImportanceSampling(**my_args).run()
    df, fig = create_plot(10)
    plt.show(fig)
