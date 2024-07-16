import numpy as np 
import scipy as sp
from pyapprox.interface.wrappers import ModelEnsemble
## Implement the work of https://arxiv.org/abs/2209.05705 in the pyapprox framework
'''
    Right hand side function with memory 
'''
class RightHandSideWithMemory:
    def __init__(self, values, b):
        self.values = values
        if callable(b):
            self.b = b
            self.mem_b = np.nan*np.ones(values.shape[1])
        else: 
            self.b = None
            self.mem_b = b.flatten()
    def __getitem__(self, ix):
        if self.b is not None:
            mask = np.isnan(self.mem_b[ix])
            if any(mask): self.mem_b[ix[mask]] = self.b(self.values[:,ix[mask]]).flatten()
        return self.mem_b[ix]
'''
    Abstract class that solves sketched least squares problem 
'''
class AbstractSketcher:
    def __init__(self, values, A, b):
        self.A = A
        self.b = RightHandSideWithMemory(values, b)
    def solve_sketched_lsq(self, sketch_sz, sketch_ix=None, scaling=None):
        if sketch_ix is None or scaling is None:
            sketch_ix, scaling = self._get_sketch_ix_and_scaling(sketch_sz)
        assert sketch_ix is not None and scaling is not None
        A_sketched, b_sketched = self._sketch_A_and_b(self.A, self.b, sketch_ix, scaling)
        xstar = np.linalg.lstsq(
                A_sketched, 
                b_sketched,
                rcond=None
            )[0]
        xstar = xstar.reshape((len(xstar),1)) # pce set coeff expects a matrix of size (L, 1)
        return xstar
    def _sketch_A_and_b(self, A, b, sketch_ix, scaling):
        A_sketched = A[sketch_ix,:] * scaling[:, np.newaxis]
        b_sketched = self.b[sketch_ix] * scaling
        return A_sketched, b_sketched
    def _get_sketch_ix_and_scaling(self, sketch_sz):
        raise NotImplementedError
'''
    Deterministic Sketching method that looks at the row space of A though the QR pivoting 
'''
class QRSketcher(AbstractSketcher):
    def _get_sketch_ix_and_scaling(self, sketch_sz):
        no_rows, no_cols = self.A.shape
        assert sketch_sz < no_rows, "Must provide sketch that is larger than the number of rows"
        assert sketch_sz < no_cols, "Assumed that we have an over determined system, that the number or columns is less than the number rows even in the sketched system"
        qr_time = int(np.ceil(sketch_sz/ no_cols))
        ix_use = np.arange(no_rows)
        sketch_ix = np.zeros(no_cols*qr_time, np.int32)
        for i in range(qr_time):
            _,_,p = sp.linalg.qr(self.A[ix_use,:].T, mode='economic', pivoting=True)
            sketch_ix[i*no_cols:(i+1)*no_cols] = ix_use[p[:no_cols]]
            ix_use = np.setdiff1d(ix_use, ix_use[p[:no_cols]])
        sketch_ix = sketch_ix[:sketch_sz]
        return sketch_ix, np.ones(len(sketch_ix))
'''
    Sample uniformly from rows
'''
class UniformSketcher(AbstractSketcher):
    def _get_sketch_ix_and_scaling(self, sketch_sz):
        no_rows_total = self.A.shape[0]
        samp_ix = np.random.choice(range(no_rows_total), size=sketch_sz, replace=True)
        unique_samp_ix, counts = np.unique(samp_ix, return_counts=True)
        real_sketch_sz = len(unique_samp_ix)
        p_unif = np.ones(real_sketch_sz) / no_rows_total
        scaling = np.sqrt(counts / (sketch_sz * p_unif))
        return unique_samp_ix, scaling 
'''
    Sample proportional to the leverage scores of rows
'''
class LeverageScoreSketcher(AbstractSketcher):
    def __init__(self, values, A, b_lofi):
        super().__init__(values, A, b_lofi)
        no_rows_total = self.A.shape[0]
        Q, _ = np.linalg.qr(self.A)
        # make the probabilty distribution be poportional to the norm of the rows for A's QR decomp
        ell = np.array([np.linalg.norm(Q[rix,:],2)**2 for rix in range(no_rows_total)])
        self.p = ell / ell.sum()
    def _get_sketch_ix_and_scaling(self, sketch_sz):
        no_rows_total = self.A.shape[0]
        samp_ix = np.random.choice(range(no_rows_total), size=sketch_sz, replace=True, p=self.p)
        unique_samp_ix, counts = np.unique(samp_ix, return_counts=True)
        real_sketch_sz = len(unique_samp_ix)
        p_unif = np.ones(real_sketch_sz) / no_rows_total
        scaling = np.sqrt(counts / (sketch_sz * p_unif))
        return unique_samp_ix, scaling 
'''
    Abstract class that leverages the boosting algorithm with bi-fidelity data 
'''
class AbstractBiFiBooster(AbstractSketcher):
    def __init__(self, values, A, model_ensemble):
        if not isinstance(model_ensemble, ModelEnsemble):
            raise TypeError(f"'model_ensemble' must be of type ModelEnsemble it is of type {type(model_ensemble)}")
        if (not 'hi' in model_ensemble.names) or (not 'lo' in model_ensemble.names):
            raise TypeError("'model_ensemble' must be a ModelEnsemble with 'lo' and 'hi' ")
        loIx = model_ensemble.names.index('lo')
        hiIx = model_ensemble.names.index('hi')
        assert loIx is not None, 'Must provide a lofi model to the model ensemble'
        assert hiIx is not None, 'Must provide a hifi model to the model ensemble'
        b_lofi = model_ensemble.functions[loIx]
        b_hifi = model_ensemble.functions[hiIx]
        self.all_ix = np.arange(values.shape[1])
        super().__init__(values, A, b_lofi)
        self.b = RightHandSideWithMemory(values, b_lofi)
        self.b_hifi = RightHandSideWithMemory(values, b_hifi)
        
    def solve_sketched_lsq(self, sketch_sz, no_trials):
        no_coeff = self.A.shape[1]
        # coefficient matrix from lofi sketches 
        IX = [None for _ in range(no_trials)]
        X = np.zeros([no_coeff, no_trials])
        # Sketch on lofi problem no_trials number of times
        for l in range(no_trials):
            IX[l], scaling = self._get_sketch_ix_and_scaling(sketch_sz)
            X[:,l] = AbstractSketcher.solve_sketched_lsq(self,sketch_sz, IX[l], scaling).flatten()
        # Select best 
        abs_err = np.array([np.linalg.norm(np.dot(self.A, X[:,l]) - self.b[self.all_ix],2) for l in range(X.shape[1])]) 
        ix_best = np.argmin(abs_err)
        # Solve sketched problem with hifi 
        bestIX_len = len(IX[ix_best])
        A_sketched, b_sketched = self._sketch_A_and_b(self.A, self.b_hifi, IX[ix_best], np.ones(bestIX_len)/ bestIX_len)
        xstar = np.linalg.lstsq(
                A_sketched, 
                b_sketched,
                rcond=None
            )[0]
        xstar = xstar.reshape((len(xstar),1)) # pce set coeff expects a matrix of size (L, 1)
        return xstar
    def _get_sketch_ix_and_scaling(self, sketch_sz):
        raise NotImplementedError
'''
    Combine the bifidelity algorithm with uniform sampling
'''
class UniformBifiBooster(AbstractBiFiBooster, UniformSketcher):
    def __init__(self, values, A, model_ensemble):
        AbstractBiFiBooster.__init__(self, values, A, model_ensemble)
    def solve_sketched_lsq(self, sketch_sz, no_trials):
        return AbstractBiFiBooster.solve_sketched_lsq(self,sketch_sz, no_trials)
    def _get_sketch_ix_and_scaling(self, sketch_sz):
        return UniformSketcher._get_sketch_ix_and_scaling(self,sketch_sz)
'''
    Combine the bifidelity algorithm with leverage scoring  
'''   
class LeverageScoreBifiBooster(AbstractBiFiBooster, LeverageScoreSketcher):
    def __init__(self, values, A, model_ensemble):
        AbstractBiFiBooster.__init__(self, values, A, model_ensemble)
    def solve_sketched_lsq(self, sketch_sz, no_trials):
        return AbstractBiFiBooster.solve_sketched_lsq(self,sketch_sz, no_trials)
    def _get_sketch_ix_and_scaling(self, sketch_sz):
        return LeverageScoreSketcher._get_sketch_ix_and_scaling(self,sketch_sz)
'''
    Given a Polynomial Chaos expansion, and a function we wish to approximate solve for coeefficients using a row sketch of the basis_matrix
'''
def fit_pce_with_sketch(pce, fun, nodes, sketch_sz, sketch_type):
    
    sketch_types={
        "qr": QRSketcher, 
        "uniform": UniformSketcher,
        "leverage_score": LeverageScoreSketcher,
    }
    sketcher = sketch_types[sketch_type](nodes, pce.basis_matrix(nodes), fun)
    coeff = sketcher.solve_sketched_lsq(sketch_sz)
    pce.set_coefficients(coeff)
    return pce 
'''
    Given a Polynomial Chaos expansion, and a model ensemble containing a low and hi fidelity  solve for coeefficients using a row sketch of the basis_matrix
'''
def fit_pce_with_bf_boosting(pce, model_ensemble, nodes, sketch_sz, no_trials, sketch_type):
    booster_types={
        "uniform": UniformBifiBooster,
        "leverage_score": LeverageScoreBifiBooster
    }
    booster = booster_types[sketch_type](
        nodes, 
        pce.basis_matrix(nodes), 
        model_ensemble
    )
    coeff = booster.solve_sketched_lsq(sketch_sz, no_trials)
    pce.set_coefficients(coeff)
    return pce 