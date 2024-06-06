# EOM-ic-MRUCC
This is a pilot implementation of the Equation-of-Motion internally contracted Multireference Coupled-Cluster (EOM-ic-MRCC) theory.

The code is based on [Forte](https://github.com/evangelistalab/forte).

## Current capabilities
- Availability of both unitary and non-unitary versions.
- Baker–Campbell–Hausdorff (BCH) expansion truncation at a specified number of commutators.
- Support for arbitrary order excitation manifold.
- Implementation of frozen core approximation.
- Single reference EOM-CC (see cas_eom.py) based on an arbitrary starting determinant.

## To-dos
- Efficient implementation
- IP, EA...
