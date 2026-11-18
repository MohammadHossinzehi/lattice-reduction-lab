"""lattice_lab: exact lattice basis reduction and its applications.

Pure Python, zero dependencies, no floating point in any decision.
"""

from .applications import (NoIntegerSolution, integer_relation, kernel_lattice,
                           minimal_polynomial, simultaneous_approximation,
                           solve_integer_system)
from .bkz import bkz, hermite_factor
from .enumeration import (babai_nearest_plane, closest_vector,
                          shortest_vector)
from .linalg import gram_determinant, gram_schmidt, is_unimodular
from .lll import (is_lll_reduced, is_size_reduced, lll, lll_integral,
                  lll_rational)
from .normal_forms import (hermite_normal_form, invariant_factors,
                           smith_normal_form)

__version__ = "1.0.0"

__all__ = [
    "lll", "lll_integral", "lll_rational", "is_lll_reduced", "is_size_reduced",
    "bkz", "hermite_factor", "shortest_vector", "closest_vector",
    "babai_nearest_plane", "hermite_normal_form", "smith_normal_form",
    "invariant_factors", "solve_integer_system", "kernel_lattice",
    "integer_relation", "minimal_polynomial", "simultaneous_approximation",
    "NoIntegerSolution", "gram_schmidt", "gram_determinant", "is_unimodular",
]
