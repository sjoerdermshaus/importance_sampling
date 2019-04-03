import numpy as np
import matplotlib.pyplot as plt


def create_and_save_plot(args, df, figsize=7, dpi=500):

    # Create a pivot table for plotting purposes
    aggfunc = {'std_true': np.sum}
    kwargs = dict(aggfunc=aggfunc,
                  index='shift',          # x-axis
                  columns='sample_size',  # legend
                  values=aggfunc.keys()   # y-axis
                  )
    df_pivot = df.pivot_table(**kwargs)

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
    fig.savefig('results.png', dpi=dpi)
    return fig
