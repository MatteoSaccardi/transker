import os
import sys
import unittest

import numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.bounds import BoundedData
from modules.kernels import cauchy_np, gaussian_np
from modules.sip import available_sip_solvers
from modules.transition import RegulatedRKTransition, SIPTransition, TransitionKernelProblem


class KernelSmokeTests(unittest.TestCase):
    def test_kernel_reference_values(self):
        self.assertAlmostEqual(cauchy_np(0.0, 0.0, 1.0), 1.0 / numpy.pi)
        self.assertAlmostEqual(gaussian_np(0.0, 0.0, 1.0), 1.0 / numpy.sqrt(2.0 * numpy.pi))


class BoundedDataSmokeTests(unittest.TestCase):
    def test_bar_delta_and_width(self):
        grid = numpy.array([0.0, 1.0])
        exact = numpy.array([1.0, 2.0])
        upper = numpy.array([1.2, 2.4])
        lower = numpy.array([0.8, 1.6])
        data = BoundedData(grid=grid, exact=exact, upper=upper, lower=lower)

        numpy.testing.assert_allclose(data.bar, numpy.array([1.0, 2.0]))
        numpy.testing.assert_allclose(data.delta, numpy.array([0.2, 0.4]))
        numpy.testing.assert_allclose(data.width, numpy.array([0.4, 0.8]))

    def test_exact_defaults_to_midpoint_when_only_bounds_are_known(self):
        grid = numpy.array([0.0, 1.0])
        upper = numpy.array([1.2, 2.4])
        lower = numpy.array([0.8, 1.6])
        data = BoundedData.from_bounds(grid=grid, upper=upper, lower=lower)

        numpy.testing.assert_allclose(data.exact, data.bar)
        numpy.testing.assert_allclose(data.bar, numpy.array([1.0, 2.0]))


class TransitionSmokeTests(unittest.TestCase):
    def make_problem(self):
        param_grid = numpy.linspace(-1.0, 1.0, 7)
        omega_grid = numpy.linspace(-2.0, 2.0, 21)
        exact = numpy.array([cauchy_np(a, 0.0, 1.0) for a in param_grid])
        data = BoundedData(
            grid=param_grid,
            exact=exact,
            upper=exact * 1.1,
            lower=exact * 0.9,
        )
        return TransitionKernelProblem(
            param_grid=param_grid,
            omega_grid=omega_grid,
            data=data,
            target_func=lambda w: cauchy_np(w, 0.0, 1.0),
            basis_func=lambda w, a: cauchy_np(w, a, 1.0),
        )

    def test_problem_vectors_and_matrix_shapes(self):
        problem = self.make_problem()
        self.assertEqual(problem.target_vector().shape, (21,))
        self.assertEqual(problem.basis_matrix().shape, (21, 7))

    def test_regulated_rk_result_is_ordered(self):
        result = RegulatedRKTransition(self.make_problem()).evaluate(alpha=1e-2)
        self.assertGreater(result.total_error, 0.0)
        self.assertGreaterEqual(result.upper, result.lower)
        self.assertEqual(result.reconstruction.shape, (21,))

    def test_regulated_rk_asymmetric_optimization_is_ordered(self):
        _, interval = RegulatedRKTransition(self.make_problem()).optimize_log_alpha(
            bounds=(-4, -1),
            grid_size=10,
        )
        self.assertEqual(interval.RK_method, "asymmetric")
        self.assertGreaterEqual(interval.upper_bound, interval.lower_bound)

    def test_sip_transition_tiny_problem(self):
        if not available_sip_solvers():
            self.skipTest("No supported CVXPY solver is installed")

        problem = self.make_problem()
        interval = SIPTransition(
            problem,
            omega_active_init=problem.param_grid,
            omega_bounds=(problem.omega_grid[0], problem.omega_grid[-1]),
            tol=1e-6,
            max_iters=5,
        ).solve_interval()

        self.assertGreaterEqual(interval.upper.rigorous, interval.lower.rigorous)
        self.assertGreaterEqual(interval.width, 0.0)


if __name__ == "__main__":
    unittest.main()
