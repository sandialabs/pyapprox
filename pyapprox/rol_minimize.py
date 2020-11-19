try:
    from _rol_minimize import *
    has_ROL = True
except:
    has_ROL = False


if __name__ == '__main__':
    np.seterr(all='raise')
    #test_TR()
    #test_rosenbrock_TR()
    test_rosenbrock_TR_constrained()
