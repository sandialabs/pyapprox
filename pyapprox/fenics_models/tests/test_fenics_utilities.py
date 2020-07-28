import unittest
# need mshr to create my own mesh and import Rectangle Circle
try:
    import dolfin as dl
    dolfin_package_missing=False
except:
    dolfin_package_missing=True

if not dolfin_package_missing:
    from pyapprox.fenics_models.tests._test_fenics_utilities import *
else:
    class TestFenicsUtilities(unittest.TestCase):
        @unittest.skip(reason="dolfin package missing")
        def test_all(self):
            pass

if __name__== "__main__":    
    fenics_utilities_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestFenicsUtilities)
    unittest.TextTestRunner(verbosity=2).run(fenics_utilities_test_suite)


    
