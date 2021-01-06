import unittest
# need mshr to create my own mesh and import Rectangle Circle
try:
    import dolfin as dl
    dolfin_package_missing = False
except:
    dolfin_package_missing = True

if not dolfin_package_missing:
    from pyapprox.fenics_models.tests._test_advection_diffusion import *
else:
    class TestTransientDiffusion(unittest.TestCase):
        @unittest.skip(reason="dolfin package missing")
        def test_all(self):
            pass

    class TestSteadyStateDiffusion(unittest.TestCase):
        @unittest.skip(reason="dolfin package missing")
        def test_all(self):
            pass

    class TestTransientAdvectionDiffusionEquation(unittest.TestCase):
        @unittest.skip(reason="dolfin package missing")
        def test_all(self):
            pass

if __name__== "__main__":
    transient_diffusion_test_suite=\
        unittest.TestLoader().loadTestsFromTestCase(TestTransientDiffusion)
    unittest.TextTestRunner(verbosity=2).run(transient_diffusion_test_suite)
    steady_state_diffusion_test_suite=\
        unittest.TestLoader().loadTestsFromTestCase(TestSteadyStateDiffusion)
    unittest.TextTestRunner(verbosity=2).run(steady_state_diffusion_test_suite)
    transient_advection_diffusion_equation_test_suite=\
        unittest.TestLoader().loadTestsFromTestCase(
            TestTransientAdvectionDiffusionEquation)
    unittest.TextTestRunner(verbosity=2).run(
        transient_advection_diffusion_equation_test_suite)

