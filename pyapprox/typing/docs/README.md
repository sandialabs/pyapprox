# PyApprox Typing Module Documentation

This directory contains developer documentation for the `pyapprox.typing` module.

## Documentation Files

### Core Conventions

1. **[ARRAY_CONVENTIONS.md](ARRAY_CONVENTIONS.md)** (18 KB)
   - Input/output array shapes
   - Jacobian shape conventions (always 2D/3D, never 1D)
   - Jacobian naming conventions (`jacobian()` vs `jacobian_wrt_params()`)
   - Random variables convention
   - When to use 1D vs 2D arrays
   - Comprehensive examples

2. **[TESTING_CONVENTIONS.md](TESTING_CONVENTIONS.md)** (18 KB)
   - Generic base class testing pattern
   - Dual backend testing (NumPy + PyTorch)
   - `load_tests` function to skip base classes
   - Backend-only code guidelines
   - Running tests with unittest
   - Complete test file template

3. **[TYPE_SAFETY_AND_LINTING.md](TYPE_SAFETY_AND_LINTING.md)** (13 KB)
   - Type annotation requirements
   - Generic types and protocols
   - mypy configuration and usage
   - ruff linting standards
   - Code style requirements
   - Type-safe patterns and examples

4. **[PROTOCOLS.md](PROTOCOLS.md)** (8.4 KB)
   - Protocol design patterns
   - Protocol composition
   - Structural subtyping

### Compliance

5. **[CONVENTIONS_COMPLIANCE_REPORT.md](CONVENTIONS_COMPLIANCE_REPORT.md)** (6.5 KB)
   - Verification that code follows array conventions
   - Evidence from actual implementation
   - Current status: ✅ 100% compliant

---

## Quick Reference

### For New Contributors

Start with these in order:

1. **[ARRAY_CONVENTIONS.md](ARRAY_CONVENTIONS.md)** - Understand input/output shapes and Jacobians
2. **[TESTING_CONVENTIONS.md](TESTING_CONVENTIONS.md)** - Learn how to write dual-backend tests
3. **[TYPE_SAFETY_AND_LINTING.md](TYPE_SAFETY_AND_LINTING.md)** - Type safety requirements

### For Implementing a New Kernel

Read:
- [ARRAY_CONVENTIONS.md](ARRAY_CONVENTIONS.md) - Jacobian shapes and naming
- [TYPE_SAFETY_AND_LINTING.md](TYPE_SAFETY_AND_LINTING.md) - Type annotations
- [TESTING_CONVENTIONS.md](TESTING_CONVENTIONS.md) - Test structure

Key points:
- `jacobian(X1, X2)` → spatial derivatives (n1, n2, nvars)
- `jacobian_wrt_params(X)` → hyperparameter derivatives (n, n, nparams)
- Both Jacobians are always multi-dimensional (never 1D)

### For Implementing a New Function

Read:
- [ARRAY_CONVENTIONS.md](ARRAY_CONVENTIONS.md) - Input (nvars, nsamples), output (nqoi, nsamples)
- [PROTOCOLS.md](PROTOCOLS.md) - FunctionProtocol interface
- [TYPE_SAFETY_AND_LINTING.md](TYPE_SAFETY_AND_LINTING.md) - Generic types

Key points:
- Use `Generic[Array]` for backend-agnostic code
- Implement `FunctionProtocol[Array]`
- `jacobian()` returns (nqoi, nvars, nsamples)

### For Writing Tests

Read:
- [TESTING_CONVENTIONS.md](TESTING_CONVENTIONS.md) - Complete guide

Key pattern:
```python
class TestMyClass(Generic[Array], unittest.TestCase):
    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

class TestMyClassNumpy(TestMyClass[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()

class TestMyClassTorch(TestMyClass[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()
```

---

## Conventions Summary

### Array Shapes

| Array Type | Shape | Notes |
|------------|-------|-------|
| Input X | (nvars, nsamples) | Samples are columns |
| Output y (training) | (nsamples, nqoi) | Samples are rows |
| Function f(X) | (nqoi, nsamples) | Transpose of y |
| Jacobian ∂f/∂X | (nqoi, nvars, nsamples) | Always multi-dim |
| Kernel jacobian | (n1, n2, nvars) | Always 3D |
| Bounds | (nparams, 2) | Always 2D |
| Hyperparameters | (nparams,) | 1D array |

### Jacobian Naming

- `jacobian()` - w.r.t. **primary variables** (inputs X)
- `jacobian_wrt_params()` - w.r.t. **hyperparameters** (θ)
- Explicit names for multi-variable classes (e.g., `jacobian_wrt_state()`)

### Type Safety

- **All** functions must have type annotations
- Use `Generic[Array]` for backend-agnostic code
- Use `Protocol` for interface definitions
- Code must pass `mypy --strict`
- Code must pass `ruff check`

### Testing

- Use generic base classes with `load_tests`
- Never use numpy/torch directly in tests
- Always test both NumPy and PyTorch backends
- Run with: `python -m unittest module.test_file`

---

## Validation

To check compliance with conventions:

```bash
# Type check
mypy pyapprox/typing/

# Lint
ruff check pyapprox/typing/

# Run all tests
python -m unittest discover -s pyapprox/typing -p "test_*.py"
```

See [CONVENTIONS_COMPLIANCE_REPORT.md](CONVENTIONS_COMPLIANCE_REPORT.md) for current compliance status.

---

## Contributing

When adding new code to `pyapprox.typing`:

1. ✅ Read relevant convention documents
2. ✅ Use type annotations throughout
3. ✅ Follow array shape conventions
4. ✅ Write dual-backend tests
5. ✅ Run mypy and ruff before committing
6. ✅ Update compliance report if adding new patterns

---

## Questions?

If you find unclear or missing documentation, please:
1. Add clarifying examples to the relevant document
2. Update the compliance report if you find violations
3. Keep documentation in sync with code

**Philosophy**: Convention documents should be the single source of truth for design patterns in the typing module.
