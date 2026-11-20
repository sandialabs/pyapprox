"""Where a discretization stack meets the coupling: each module builds
subdomain responses from one stack's objects.

The coupling core imports no stack, and no stack imports the coupling;
only these modules import both. Import them by module path: this package
re-exports nothing, so importing it loads no stack.
"""
