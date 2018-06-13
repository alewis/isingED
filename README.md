# Exact diagonalization code for 2D Ising model

Simple code to generate ground state wavefunctions of the 2D Ising model
using (Lancoz) exact diagonalization. The nearest-neighbour Hamiltonian
is applied sequentially to each coupling in the lattice. This is done by 
noting that the full Hamiltonian is a sum of operator strings such as 
H = 1 * 1 * 1 * h * 1... + 1 * 1 * 1 * 1 * h...
We only actually store h*1 +  1*h, and perform the operation H*v by applying
h*v, which implements one of the  operator strings, then cyclically permuting
v so that h*v now implements the next one. This is done until all the cyclic
permutations are exhausted. Periodic boundary conditions and translation
invariance are both assumed.

For a 1D lattice, i.e.
```
_ _ _ _ 
1 2 3 4
```
we need to apply the permutation (1 2 3 4) for a full cycle: i.e.
the action of the local hamiltonian on each of
[1 2 3 4],  [4 1 2 3], [3 4 1 2], and [2 3 4 1] needs to be summed. For a
2D lattice, say
```
_ _ _
1 4 7
_ _ _
2 5 8
_ _ _
3 6 9
```

(note Julia arrays are column-major), we need to do this for each row
and once for each column of the lattice:
i.e. (123)(4)(5)(6)(7)(8)(9) for a full cycle, then
(456)..., then (789), then (147), (456), and (789).
The present code handles this by first generating all of the necessary permutations
in the arrays permuteidxs and permutevals, then applying a full cycle of each
one.

The wavefunction is then saved to disk as an array with shape 
(2, 2, 2, ... N). Each index runs over the two spin states of a particular
site. 

As a quick test, with hfield=0 the energy density should be -J*dim, where 
dim is the dimensionality of the lattice.

Unlike Morningstar's
code, no effort is made to find a compressed basis using symmetry structures 
(e.g. the wavefunction has 2^N elements). This makes the code much easier
to write, but considerably less efficient; about 16 lattice sites is probably
the most it can handle.

This code is a slight modification of Markus Hauru's implementation,
which covers the 1D case. That code can be found at
https://github.com/mhauru/MPS-in-Julia-minicourse/blob/master/meeting_01.ipynb.

### Dependencies
To install dependencies, run `dependencies.jl`:
```
julia dependencies.jl
```
If errors occur, try updating Julia to v0.6.3 (the version in the Ubuntu repos did *not* work) 
