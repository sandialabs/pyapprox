import unittest
# need mshr to create my own mesh and import Rectangle Circle
try:
    import dolfin as dl
    dolfin_package_missing=False
except:
    dolfin_package_missing=True

if not dolfin_package_missing:
    from pyapprox.fenics_models.tests._test_helmholtz import *
else:
    class TestHelmholtz(unittest.TestCase):
        @unittest.skip(reason="dolfin package missing")
        def test_all(self):
            pass
        
if __name__== "__main__":
    #dl.set_log_level(40)
    helmholtz_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestHelmholtz)
    unittest.TextTestRunner(verbosity=2).run(helmholtz_test_suite)

