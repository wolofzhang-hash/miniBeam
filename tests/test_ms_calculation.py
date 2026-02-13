import unittest
import numpy as np

from minibeam.core.pynite_adapter import compute_ms_from_internal_forces


class TestMSCalculation(unittest.TestCase):
    def test_typical_case_matches_document_example(self):
        result = compute_ms_from_internal_forces(
            N=np.array([20000.0]),
            Mz=np.array([8_000_000.0]),
            My=np.array([1_500_000.0]),
            T=np.array([600_000.0]),
            area=2500.0,
            Iz=8_000_000.0,
            Iy=3_000_000.0,
            J=500_000.0,
            c_z=75.0,
            c_y=40.0,
            sigma_allow=345.0 / 1.5,
            shape_factor_z=1.12,
            shape_factor_y=1.05,
            shape_factor_t=1.20,
        )

        self.assertAlmostEqual(float(result["sigma"][0]), 77.62060, places=4)
        self.assertAlmostEqual(float(result["tau_t"][0]), 75.0, places=6)
        self.assertAlmostEqual(float(result["margin_elastic"][0]), 0.293218, places=4)
        self.assertAlmostEqual(float(result["margin"][0]), 0.519884, places=4)


    def test_bending_components_do_not_cancel_each_other(self):
        result = compute_ms_from_internal_forces(
            N=np.array([0.0]),
            Mz=np.array([1_000_000.0]),
            My=np.array([-1_000_000.0]),
            T=np.array([0.0]),
            area=1000.0,
            Iz=1_000_000.0,
            Iy=1_000_000.0,
            J=1_000_000.0,
            c_z=10.0,
            c_y=10.0,
            sigma_allow=100.0,
        )

        self.assertAlmostEqual(float(result["sigma"][0]), np.sqrt(200.0), places=6)

    def test_shape_factor_lower_than_one_is_clamped(self):
        base = compute_ms_from_internal_forces(
            N=np.array([0.0]),
            Mz=np.array([1_000_000.0]),
            My=np.array([0.0]),
            T=np.array([0.0]),
            area=1000.0,
            Iz=1_000_000.0,
            Iy=1_000_000.0,
            J=1_000_000.0,
            c_z=10.0,
            c_y=10.0,
            sigma_allow=100.0,
            shape_factor_z=1.0,
            shape_factor_y=1.0,
            shape_factor_t=1.0,
        )
        clamped = compute_ms_from_internal_forces(
            N=np.array([0.0]),
            Mz=np.array([1_000_000.0]),
            My=np.array([0.0]),
            T=np.array([0.0]),
            area=1000.0,
            Iz=1_000_000.0,
            Iy=1_000_000.0,
            J=1_000_000.0,
            c_z=10.0,
            c_y=10.0,
            sigma_allow=100.0,
            shape_factor_z=0.5,
            shape_factor_y=0.5,
            shape_factor_t=0.5,
        )

        self.assertTrue(np.allclose(base["sigma"], clamped["sigma"]))
        self.assertTrue(np.allclose(base["tau_t"], clamped["tau_t"]))
        self.assertTrue(np.allclose(base["margin"], clamped["margin"]))


if __name__ == "__main__":
    unittest.main()
