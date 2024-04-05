# My_EOM_MRCC
This is a pilot implementation of the Equation-of-Motion internally contracted Multireference Coupled-Cluster (EOM-ic-MRCC) theory.
This implementation is based on [Forte](https://github.com/evangelistalab/forte).

## Current capabilities
- Both unitary and non-unitary versions are available.
- Baker–Campbell–Hausdorff expression can be truncated at a certain number of commutators.
- Supports arbitrary order excitation manifold.
- Frozen core approximation.
- Single reference EOM-CC (see cas_eom.py) based on an arbitrary starting determinant.

## To-dos
- Efficient implementation
- IP, EA...
