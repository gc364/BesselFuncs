# PyTorch General Bessel Functions
## Aims
The aim of this repository is to provide a library of Bessel functions of complex argument and real order, that are compatible with PyTorch autograd.
The methods of computation are heavily inspired by the FORTRAN AMOS and SLATEC libraries.
## Contents
The module contains functions to calculate the following:
1. Ordinary Bessel function of the first kind for complex argument and Real order
2. Ordinary Bessel function of the second kind for complex argument and Real order
3. Modified Bessel function of the first kind for complex argument and Real order
4. Modified Bessel function of the second kind for complex argument and Real order
5. Hankel functions of the first and second kinds for complex argument and Real order
6. Derivatives of the Bessel functions.
## Note
Only 1. above have explicit functions for finding the derivatives. However, a slightly modified version of the SciPy function for doing this is 
included, which describes the phase argument needed to apply to other types.