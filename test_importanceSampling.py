from unittest import TestCase
import numpy as np
from main import ImportanceSampling as ImSa
from scipy.stats import norm


class TestImportanceSampling(TestCase):
    def setUp(self):
        self.data = np.array([5, 2, 4, 3, 1, 6])
        self.likelihood_ratios = [None, np.ones((len(self.data),))]

        # 1 -> 0, 2 -> 20, 3 -> 40, 4 -> 60, 5 -> 80, 6 -> 100

    def test_percentile_regular_exact(self):
        for likelihood_ratio in self.likelihood_ratios:
            # Exact percentiles
            self.assertEqual(ImSa.percentile(self.data,   0, likelihood_ratio), 1)
            self.assertEqual(ImSa.percentile(self.data,  20, likelihood_ratio), 2)
            self.assertEqual(ImSa.percentile(self.data,  80, likelihood_ratio), 5)
            self.assertEqual(ImSa.percentile(self.data, 100, likelihood_ratio), 6)

    def test_percentile_regular_nearest(self):
        for likelihood_ratio in self.likelihood_ratios:
            # Nearest percentiles
            self.assertEqual(ImSa.percentile(self.data,  5, likelihood_ratio), 1)
            self.assertEqual(ImSa.percentile(self.data, 15, likelihood_ratio), 2)
            self.assertEqual(ImSa.percentile(self.data, 85, likelihood_ratio), 5)
            self.assertEqual(ImSa.percentile(self.data, 95, likelihood_ratio), 6)

    def test_percentile_regular_midpoint(self):
        for likelihood_ratio in self.likelihood_ratios:
            # Mid-point
            self.assertEqual(ImSa.percentile(self.data, 10, likelihood_ratio), 1)
            if likelihood_ratio is None:
                # Potential bug? Approaching from the left or right should not matter...
                self.assertEqual(ImSa.percentile(self.data, 90, likelihood_ratio), 5)
            else:
                self.assertEqual(ImSa.percentile(self.data, 90, likelihood_ratio), 6)

    def test_percentile_regular_50(self):
        for likelihood_ratio in self.likelihood_ratios:
            self.assertEqual(ImSa.percentile(self.data, 50, likelihood_ratio), 3)

    def test_percentile_IS_left_tail(self):
        shift = -2
        shifted_data = self.data + shift
        p = norm.pdf(shifted_data, loc=3.5, scale=1)
        q = norm.pdf(shifted_data, loc=1.5, scale=1)
        likelihood_ratio = p / q
        results = [[0.01, -1], [0.1, 0], [0.14, 0], [1.1, 1], [7, 2], [49, 3], [50, 3]]
        for r in results:
            self.assertEqual(ImSa.percentile(shifted_data, r[0], likelihood_ratio), r[1])

    def test_percentile_IS_right_tail(self):
        shift = 3
        shifted_data = self.data + shift
        p = norm.pdf(shifted_data, loc=3.5, scale=1)
        q = norm.pdf(shifted_data, loc=6.5, scale=1)
        likelihood_ratio = p / q
        results = [[99.999, 9], [99.998, 8], [99.997, 8], [99.96, 7], [98, 6], [60, 5]]
        for r in results:
            self.assertEqual(ImSa.percentile(shifted_data, r[0], likelihood_ratio), r[1])

    def tearDown(self):
        pass
