import unittest

from ordi.eval.state_model_fit import _errors


class StateModelFitTest(unittest.TestCase):
    def test_errors(self):
        mae, rmse = _errors([1.0, 3.0], [2.0, 1.0])
        self.assertAlmostEqual(mae, 1.5)
        self.assertAlmostEqual(rmse, (2.5) ** 0.5)

    def test_errors_reject_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            _errors([1.0], [])


if __name__ == "__main__":
    unittest.main()
