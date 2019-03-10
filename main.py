import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp


class ImportanceSampling:
    def __init__(self, quantile, sample_sizes=10000, shifts=3, sim_sizes=1, pool_size=1):
        # Initialize arguments
        self.quantile = quantile
        self.sample_sizes = [sample_sizes] if isinstance(sample_sizes, int) else sample_sizes
        self.shifts = [shifts] if isinstance(shifts, (int, float)) else shifts
        self.sim_sizes = [sim_sizes] if isinstance(sim_sizes, int) else sim_sizes
        self.pool_size = pool_size

        # Calculate the true percentile
        self.truth = norm.ppf(self.quantile / 100.0)

        # Output columns
        self.columns = ['sample_size',
                        'shift',
                        'sim_size',
                        'quantile',
                        'truth',
                        'mean',
                        'std_mean',
                        'std_true',
                        'min',
                        'max',
                        'time']

    @staticmethod
    def percentile(data, quantile, likelihood_ratio=None):
        """
         Using the 'nearest' interpolation method, PERCENTILE will give the desired percentile
         from the DATA at the supplied QUANTILE.
         If LIKELIHOOD_RATIO is None, then the equally weighted/regular percentile is returned.
         If LIKELIHOOD_RATIO is not None, then the IS inspired likelihood ratio is used to derive
         a weighted percentile.
         Note that None and an array of ones will both give the regular percentile.

        :param data: numpy array
        :param quantile: quantile as percentage, e.g. 90 for the 90% percentile.
        :param likelihood_ratio: likelihood ratio evaluated at the data points
        :return: percentile at the desired quantile.
        """

        if likelihood_ratio is None:
            return np.percentile(data, quantile, interpolation='nearest')
        else:
            sample_size = len(data)
            # likelihood ratio divided by N - 1
            lr = likelihood_ratio / (sample_size - 1)
            idx = data.argsort()

            if quantile > 50:  # Right tail
                lr[np.argmax(data)] = 0
                lr_cumsum = np.flip(np.flip(lr[idx]).cumsum())
                tail_probability = 1.00 - quantile / 100.0
            else:              # Left tail
                lr[np.argmin(data)] = 0
                lr_cumsum = lr[idx].cumsum()
                tail_probability = quantile / 100.0
            idx_nearest = np.argmin(abs(lr_cumsum - tail_probability))
            return data[idx[idx_nearest]]

    def generate_importance_sample_and_calculate_percentile(self, sample_size, shift, sim_number):
        """
        Generate one importance sample and calculate the percentile.

        :param sample_size: size of the sample
        :param shift: mean shift used to simulate the importance sample
        :param sim_number: set the seed for random number generator
        :return: simulated percentile
        """

        # Set the seed based on sim_number for parallel computing (reproducibility)
        np.random.seed(sim_number)

        # Generate the importance sample by adding (mean) shift (also called translation)
        shifted_r = np.random.normal(size=(sample_size,)) + shift

        # Calculate p, q and the likelihood ratio
        p = norm.pdf(shifted_r, loc=0, scale=1)
        q = norm.pdf(shifted_r, loc=shift, scale=1)
        likelihood_ratio = p / q

        # Calculate and return the percentile
        return self.percentile(shifted_r, self.quantile, likelihood_ratio)

    def process_sim_results(self, sim_percentiles, sample_size, shift, sim_size, sim_time):
        """
        SIM_PERCENTILES contains estimated percentiles for SIM_SIZE generated IS samples. This function calculates some
        useful statistics to assess the performance of IS.
        :param sim_percentiles: a list of size SIM_SIZE of simulated IS percentiles
        :param sample_size: size of the IS samples underlying the estimated percentiles
        :param shift: (mean) shift used for generating IS samples
        :param sim_size: number of generated IS samples
        :param sim_time: time in seconds spend on generating the IS samples and calculating percentiles
        :return: a list with statistics
        """
        return [sample_size,
                shift,
                sim_size,
                self.quantile,
                self.truth,
                np.mean(sim_percentiles),
                np.std(sim_percentiles, ddof=1),
                np.sqrt(np.sum((sim_percentiles - self.truth) ** 2) / (sim_size - 1)),
                np.min(sim_percentiles),
                np.max(sim_percentiles),
                sim_time]

    def run(self):
        """
        Importance Sampling loop over 1) sample sizes,
                                      2) (mean) shifts and
                                      3) sim sizes
        :return: DataFrame with results
        """
        results = []
        for sample_size in self.sample_sizes:
            for shift in self.shifts:
                for sim_size in self.sim_sizes:

                    # Prepare arguments for starmap
                    iterable = ((sample_size, shift, i) for i in range(sim_size))
                    chunk_size = int(sim_size / self.pool_size)

                    # Start the simulation
                    start = time.time()
                    with mp.Pool(self.pool_size) as pool:
                        sim_percentiles = pool.starmap(self.generate_importance_sample_and_calculate_percentile,
                                                       iterable=iterable,
                                                       chunksize=chunk_size)
                    sim_time = round(time.time() - start, 4)

                    # Display some information
                    print(f'Sample size: {sample_size}, Shift: {shift}, Sim size: {sim_size}, Sim time: {sim_time}')
                    results.append(self.process_sim_results(np.array(sim_percentiles),
                                                            sample_size,
                                                            shift,
                                                            sim_size,
                                                            sim_time))

        # Collect the result in a DataFrame
        df = pd.DataFrame(results, columns=self.columns)
        df.to_excel('results.xlsx')
        return df


def main():
    sample_sizes = [int(5e3), int(1e4), int(5e4), int(1e5), int(5e5), int(1e6)]
    sim_sizes = 1000
    args = dict(quantile=99.95,
                sample_sizes=sample_sizes,
                shifts=np.linspace(0, 6, 13),
                sim_sizes=sim_sizes,
                pool_size=10)
    df = ImportanceSampling(**args).run()

    # Create a pivot table for plotting purposes
    aggfunc = {'std_true': np.sum}
    index = 'shift'          # x-axis
    columns = 'sample_size'  # legend
    values = aggfunc.keys()  # y-axis
    df_pivot = df.pivot_table(values=values, index=index, columns=columns, aggfunc=aggfunc)

    # Create a plot which displays the precision introduced by IS
    fig, ax = plt.subplots()
    ax.plot(df_pivot, '-o')
    ax.set_xlabel('Mean shift')
    ax.set_ylabel(f'Standard deviation (based on {sim_sizes} samples)')
    ax.set_title('Importance Sampling for various mean shifts and sample sizes')
    ax.set_ylim([0, 0.02])
    ax.grid(b=True)
    ax.legend(labels=sample_sizes, title='Sample size')
    fig.savefig('results.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
