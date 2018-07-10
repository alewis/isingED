import numpy as np
from copy import copy, deepcopy
import numpy.linalg as npla
from scipy import linalg as spla
from random import random
import scipy.sparse.linalg
import itertools
import collections

#******************************************************************************
#Clifford gates
#******************************************************************************
Sig_x = np.array([ [0+0.j, 1+0.j],
                   [1+0.j, 0+0.j]] )

Sig_y = np.array([ [0, -1.j],
                   [1.j, 0]] )

Sig_z = np.array([ [1,  0+0.j],
                   [0, -1]] )

hadamard_gate = (1./np.sqrt(2))*np.array([[ 1.+0.j, 1.+0.j],
                                          [1.+0.j, -1.+0.j] ])

phase_gate =  np.array([ [1.+0.j, 0.+0.j],
                         [0.+0.j, 0.+1.j] ])

HSdag = np.dot(hadamard_gate, np.conj(phase_gate.T))

basisdict = { 'X' : hadamard_gate,
              'Y' : HSdag,
              'Z' : None
            }

def pauliprojector(direction):
    direction = np.array(direction)
    direction /= npla.norm(direction)
    pvec = direction[0] * Sig_x + direction[1] * Sig_y + direction[2] * Sig_z
    pvec = 0.5*(np.eye(2) + pvec)
    return pvec

#*****************************************************************************
#ModelWavefunctions
#*****************************************************************************
def optochain(n, op, N):
    d = op.shape[0]
    if op.shape != (d, d):
        raise ValueError("op had bad shape ", op.shape)
    chain = [np.eye(d)]*N
    chain[n] = op
    opchain = OperatorChain(chain)
    return opchain


def makeisingground(Nx, Ny, J, h):
    sparseham = makeisingham(Nx, Ny, J, h, sparse=True)  
    E, wfdata = sparseham.eigs(k=1)
    wf = SpinChainWavefunction(wfdata, Nx, Ny)
    return E, wf


class SpinChainWavefunction(object):
    """Represents a wavefunction on a Hilbert space
       H = H1^d \otimes H2^d \otimes H3^d ... H^Nd.
       The coefficients are stored in self.wf which has shape self.shape.
    """
    def __init__(self, wf, Nx, Ny):
        self.shape = wf.shape
        self.wf = np.array(wf)
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx*Ny
        self.d=2
        if self.wf.size != self.d**self.N:
            print "wf had wrong size; wf.size, d, N: ", wf.size, d, N 
    
    def cascadedmeasurement(self, basis, directional=False):
        """Simulates a sequence of measurements at each site of the lattice.
           'basis' may take one of two forms depending on the value of
           'directional'. If 'directional' is False, 'basis' is a string
           or list
           corresponding to the local Pauli basis in which to measure at
           each site; e.g. 'XXYZ' would first measure IIIZ, then IIYI, then
           'IXII', etc. If 'directional' is True, 'basis' is a list of 
           three-vectors corresponding to the spin measurement direction
           at the appropriate site. An array of integers (either +1 or -1)
           is returned, each one representing the simulated outcome at 
           the corresponding lattice site.
        """

        basis = list(basis)
        if len(basis) != self.N:
            raise ValueError("Basis needs one operator per site.")
        outcomelist = self.__docascadedmeasurement(basis, directional)
        outcomearr = np.array(outcomelist).reshape((self.Ny, self.Nx))
        return outcomearr

    def __docascadedmeasurement(self, basis, directional, outcome=None):
        if outcome is None:
            outcome = collections.deque()
        thissite = len(basis)-1
        if not directional:
            thisoutcome, projectedwf = self.measure(thissite, basis.pop())
        else:
            thisoutcome, projectedwf = self.measurespin(thissite, basis.pop())
        outcome.appendleft(thisoutcome)
        if len(outcome) == self.N:
            return list(outcome)
        else:
            return projectedwf.__docascadedmeasurement(basis, directional, 
                                                     outcome=outcome)

    def measurespin2d(self, x, y, direction):
        return self.measurespin(self.__linearindex(x, y), direction)

    def measurespin(self, site, direction):
        """
        Simulate measurement of the spin in direction 'direction' (a container
        with 3 elements) at site 'site'.
        """
        direction = np.array(direction)
        direction /= npla.norm(direction)
        spin_plus = pauliprojector(direction)
        spin_minus = pauliprojector(-1.0*direction)
        pplus = self.expectationvalue(site, spin_plus)
        pminus = self.expectationvalue(site, spin_minus)
        if abs(1- pplus - pminus) > 1E-13:
            raise ValueError("Something went wrong; pplus, pminus = ", pplus, 
                             pminus)
        toss = random()
        if toss < pplus:
            projector = spin_plus
            outcome = 1
        else:
            projector = spin_minus
            outcome = -1
        newpsi = self.applygate(site, projector).normalize()
        return outcome, newpsi
    
    def measure2d(self, x, y, basis='Z'):
        return self.measure(self.__linearindex(x, y), basis=basis)


    def measure(self, site, basis='Z'):
        """
        Simulates a measurement upon the 'site'th qbit in the Pauli basis
        'basis', which is the computational (Z) basis by default. An error
        is thrown unless basis is either 'X', 'Y', or 'Z'. The basis is applied
        by applying the relevant local unitary projecting into its eigenspace.

        For a three-qubit chain the amplitudes appear in linear memory
        in the order
        |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>.
                       |-------|
                         glen=2
        Suppose we wish to measure the second qubit. Then we need to compute
        the two summed amplitudes of all the terms which have the same value
        at the second site. 

        Our strategy is to permute the coefs into a 2**(N-1) x 2 array so
        that each outcome lies on its own row. Then the relevant amplitudes
        are just the sum of their respective rows, and simulating the measurement
        comes down to simulating a coin toss. The state after measurement is
        just the row corresponding to the outcome we obtained, divided by
        the appropriate normalization from the projection postulate.
                      2
                |  outcome +1 (|a0b>) |
       2**(N-1) |  outcome -1 (|a1b>) |

        """
        if site>=self.N:
            raise IndexError("Site too large; site, N: ", site, self.N)
        try:
            thisgate = basisdict[basis]
        except KeyError:
            raise ValueError("Invalid basis choice ", basis)
        projectedwf = self.applygate(site, thisgate)
        N = projectedwf.N
    
        glen = 2**(N-1-site) 
        coefs = projectedwf.wf
        #coefs2 = deepcopy(coefs)
        coefs = coefs.reshape([2]*N)
        coefs = np.swapaxes(coefs, 0, site)
        coefs = coefs.reshape((2, 2**(N-1))) 
        
        #print "before (new): \n", coefs
        up = coefs[0, :]
        Pup = np.sum(np.vdot(up, up))
        down = coefs[1, :]
        Pdown = np.sum(np.vdot(down, down))
        
        if abs(1- Pdown - Pup) > 1E-13:
            raise ValueError("Something went wrong; Pdown, Pup = ", Pdown, Pup)

        toss = random()
        if toss < Pup:
            coefs[0, :] /= np.sqrt(Pup)
            coefs[1, :] = 0.+0.j
            outcome = 1
            #projectedwf = SpinChainWavefunction(up, self.Nx, self.Ny)
        else:
            coefs[0, :] = 0.+0.j
            coefs[1, :] /= np.sqrt(Pdown)
            outcome = -1
            #projectedwf = SpinChainWavefunction(down, self.Nx, self.Ny)
        # coefs2 = deepcopy(coefs)
        # if glen != 1:
            # coefs2 = coefs2.reshape([glen]*Ngroups)
            # coefs2 = coefs2.transpose(permutation)
        # coefs2 = coefs2.reshape((coefs2.size))
        
        coefs = coefs.reshape([2]*N)
        coefs = np.swapaxes(coefs, 0, site)
        coefs = coefs.reshape((coefs.size))
        #print "after (new): \n", coefs
        #print "after (old): ", coefs2
        projectedwf.wf = coefs #this may be unnecessary

        return outcome, projectedwf

    def __linearindex(self, x, y):
        return y*self.Nx + x


    def expectationvalue(self, n, op):
        """
        <psi| I1 \otimes I2 \otimes...op_n... I_N |psi>
        """
        opchain = optochain(n, op, self.N)
        return self.expectationvaluechain(opchain)
    
    def expectationvalue2d(self, x, y, op):
        """
        <psi| I1 \otimes I2 \otimes...op_n... I_N |psi>
        with n appropriately calculated from lattice coords
        """
        n = self.__linearindex(x, y)
        return self.expectationvaluesite(n, op)

    def expectationvaluechain(self, opchain):
        """
        <psi| Op1 \otimes Op2 \otimes... Op_n |psi>
        """
        return self.overlap(self.applychain(opchain)).real

    def applygate(self, n, op):
        """
        |psi'> = I_1 \otimes I_2 ... op_n ... I_N |psi>
        """
        if op is None:
            return SpinChainWavefunction(deepcopy(self.wf), self.Nx, self.Ny)
        opchain = optochain(n, op, self.N)
        return self.applychain(opchain)

    def applygate2d(self, x, y, op):
        """
        <psi| I1 \otimes I2 \otimes...op_n... I_N |psi>
        with n appropriately calculated from lattice coords
        """
        n = self.__linearindex(x, y)
        return self.applygate(n, op)

    def applychain(self, opchain):
        """
        |psi'> = Op1 \otimes Op2 \otimes... Opn |psi>
        The ops are an OperatorChain. 
        """

        opvec = opchain.timesvector(self.wf)
        return SpinChainWavefunction(opvec, self.Nx, self.Ny)

    def norm(self):
        """
        <\psi|\psi>
        """
        return npla.norm(self.wf)

    def overlap(self, other):
        """
        <\phi | \psi>
        """
        return np.vdot(other.wf, self.wf)

    def normalize(self):
        newwf = self.wf / self.norm()
        return SpinChainWavefunction(newwf, self.Nx, self.Ny)



#*****************************************************************************
#Nearest Neighbour Hamiltonians
#*****************************************************************************
class NNHamiltonian(object):
    def __init__(self, Nx, Ny, leftterms, rightterms, coefs, d=2):
        self.Nx = Nx
        self.Ny = Ny
        self.shape = (d**(Nx*Ny), d**(Nx*Ny))
        self.d=d
    
    def matvec(self, vec):
        """
        Return H * vec.
        """
        pass

    def vecmat(self, vec):
        """
        Return vec*H.
        """
        pass

    def aslinearoperator(self):
        pass

    def eigs(self, k):
        """
        Return the first k lowest-energy eigenvalues and eigenstates.
        """
        pass

class NNHamiltonianDense(NNHamiltonian):
    def __init__(self, Nx, Ny, leftterms, rightterms, coefs, d=2):
        """Build from three lists: leftterms, rightterms, and coefs. 
           Each of these represents a term of the form
           coefs[n] * Sum_<ij> (leftterms[n]_i rightterms[n]_j). 
        """
        if Nx*Ny > 14:
            raise ValueError("Cannot use dense Hamiltonian for ", Nx*Ny, "sites.")

        super(NNHamiltonianDense, self).__init__(Nx, Ny, leftterms, rightterms, 
                                                 coefs, d=d)
        self.data = np.zeros(self.shape, dtype=np.complex128)
        for left, right, coef in itertools.izip(leftterms, rightterms, coefs):
            thesedata = np.zeros(self.shape, dtype=np.complex128)
            for y, x in itertools.product(range(0, Ny), range(0, Nx)):
                thesedata += self.__maketerm(x, y, left, right) 
            self.data += coef*thesedata

    def __maketerm(self, sitex, sitey, siteterm, nextterm):
        """
        Return the summed terms in the 2D nearest-neighbour Hamiltonian 
        H = Sum_<ij> coef * (siteterm_i \otimes nextterm_j) for which
        i refers to the lattice site (sitex, sitey). 
        """
        rightchain, downchain = nnchain2d(sitex, sitey, self.Nx, self.Ny, 
                                          siteterm, nextterm, d=self.d)
        rightterm = 0.j
        downterm = 0.j
        if rightchain is not None:
            rightterm = rightchain.todense()
        if downchain is not None:
            downterm = downchain.todense()
        return rightterm + downterm

    def matvec(self, vec):
        """ return H * vec
        """
        return np.dot(self.data, vec)

    def vecmat(self, vec):
        """ return vec.T * H
        """
        return np.dot(vec, self.data)

    def aslinearoperator(self):
        return scipy.sparse.linalg.LinearOperator(self.shape, self.matvec)

    def eigs(self, k="all"):
        """The first k eigenvalues and normalized eigenvectors.
        """
        w, v = npla.eigh(self.data)
        if k=="all":
            return w, v
        w = w[:k]
        v = v[:, :k]
        return w, v
    

class NNHamiltonianSparse(NNHamiltonian):
    def __init__(self, Nx, Ny, leftterms, rightterms, coefs, d=2):
        """Build from three lists: leftterms, rightterms, and coefs. 
           Each of these represents a term of the form
           coefs[n] * Sum_<ij> (leftterms[n]_i rightterms[n]_j). 
        """
        super(NNHamiltonianSparse, self).__init__(Nx, Ny, leftterms, 
                                                  rightterms, coefs, d=d)
                
        self.chains = []
        for left, right, coef in itertools.izip(leftterms, rightterms, coefs):
            for y, x in itertools.product(range(0, Ny), range(0, Nx)):
                self.chains += self.__maketerm(x, y, left, right, coef) 

    def __maketerm(self, sitex, sitey, siteterm, nextterm, coef):
        term = []
        right, down = nnchain2d(sitex, sitey, self.Nx, self.Ny, 
                                siteterm, nextterm,
                                coef=coef, d=self.d)
        if right is not None:
            term += [right,]
        if down is not None:
            term += [down,]
        return term

    def densedata(self):
        out = 0.0
        for chain in self.chains:
            out += chain.todense()
        return out

    def matvec(self, vec, shift=0):
        """ return (H-shift*I) * vec.
            Adding a shift is useful for iterative eigensolvers, which converge
            most efficiently to the eigenvalue of largest magnitude. In some
            cases without the shift this can be a highly excited state (e.g.
            with large positive energy) rather than the ground state. In such
            cases it is useful to "shift" the entire spectrum so that the
            ground state energy has the largest magnitude. 
        """
        output = np.zeros(vec.shape, dtype=np.complex128)
        for chain in self.chains:
            output += chain.timesvector(vec)
        output -= shift*vec
        return output

    def aslinearoperator(self, shift=0):
        shiftmatvec = lambda(vec): self.matvec(vec, shift=shift)
        return scipy.sparse.linalg.LinearOperator(self.shape, shiftmatvec)

    def eigs(self, k=2, shift="auto"):
        """The first k eigenvalues and normalized eigenvectors.
           The eigensolver finds the k-dominant (largest magnitude) 
           eigenvalues/vectors of H-shift*I. We then add 'shift' back to the
           reported eigenvalues. This is done so that the ground state
           has the largest magnitude (adding a multiple of the identity does not 
           affect the eigenvectors).

           If shift is unspecified, we make the guess 100*Nx*Ny, which usually
           works.
        """
        if shift == "auto":
            #shift = 0.
            shift = (100.+0.j)*self.Nx * self.Ny
        op = self.aslinearoperator(shift=shift)
        failedonce = False
        try:
            w, v = scipy.sparse.linalg.eigsh(op, k=k)
        except:
            if not failedonce:
                w, v = scipy.sparse.linalg.eigsh(op, k=k, ncv=max(2*k+1, 20))
                failedonce = True
            else:
                raise ValueError("ARPACK failed to converge.")

        w = w + shift.real
        v /= npla.norm(v, axis=0)
        w = w.reshape((len(w), 1))
        joined = np.concatenate((w, v.T), axis=1)
        joinedsorted = joined[joined[:,0].argsort()]
        wsorted = joinedsorted[:,0].real
        vsorted = joinedsorted[:,1:].T
        return wsorted, vsorted

def nnhamfactory(Nx, Ny, lefts, rights, coefs, d=2, sparse=False):
    if sparse:
        return NNHamiltonianSparse(Nx, Ny, lefts, rights, coefs, d=d)
    else:
        return NNHamiltonianDense(Nx, Ny, lefts, rights, coefs, d=d)

def makeisingham(Nx, Ny, J, h, sparse=False):
    lefts = [Sig_x, Sig_z]
    rights = [Sig_x, np.eye(2)]
    coefs = [-J, -h]
    return nnhamfactory(Nx, Ny, lefts, rights, coefs, sparse=sparse)
        # self.J = J
        # self.h = h
        # lefts = [Sig_x, np.eye(2), Sig_z]
        # rights = [Sig_x, Sig_z, np.eye(2)]
        # coefs = [-J, -h/2.0, -h/2.0]
        # super(IsingWavefunction, self).__init__(Nx, Ny, lefts, rights, coefs,
                                                # state=state, sparse=sparse, d=d)

#******************************************************************************
#OperatorChain
#******************************************************************************

def vec(mat):
    """The 'vec' operator which appears in discussions of the Kronecker product.
       The transpose accounts for numpy's row-major storage.
    """
    return np.ravel(mat.T)

class OperatorChain(object):
    """
    A chain C of operators of the form 
    C = c0 \otimes c1 \otimes...

    It is internally stored as a list [c0, c1, c2...], where each c0
    is a dxd matrix. Such a list is to be fed to the constructor.
    You can convert c_start : c_end to their dense representation using
    the method todense(start, end). 

    Method timesvector(psi) returns C*psi. Due to certain properties of the
    Kronecker product, only the dense matrices c0:cN/2 and cN/2:cN (i.e.
    the left and right halves of the chain; in the case N is odd the appropriate
    rounding is done) need be 
    computed to do this. timesvectordense(psi) does the same thing by
    computing the full matrix C; this is provided to test timesvector.
    
    __storehalves() computes the dense matrices c0:cN/2 and cN/2:cN and stores
    the results (the former is transposed) as members. 
    This is done the first time timesvector(psi)
    is called (or __storehalves explicitly), and never again.
    """
    def __init__(self, chain):
        self.chain = chain
        self.N = len(chain)
        if self.N < 2:
            raise ValueError("OperatorChain must have at least length 2")
        self.d = chain[0].shape[0]
        if not all([link.shape==(self.d,self.d) for link in chain]):
            raise ValueError("chain had bad shape: ", [link.shape for link in chain])
        self.halvesready = False

    def __storehalves(self):
        if not self.halvesready:
            halfN = int(round(self.N/2))
            BT = self.todense(end=halfN).T
            A = self.todense(start=halfN)
            self.leftT = BT
            self.right = A
            self.halvesready = True

    def timesvector(self, psi):
        """
        This function does a sparse computation of H |psi>, where 
        H = chain[0] \otimes chain[1] \otimes chain[2] ...
        Chain is a length-N list of dxd matrices and psi is a d**N vector. 

        It exploits the identity 
        (B.T \otimes A) |psi> = vec( A \Psi B ), where
        H = A\otimesB and |psi> = vec(\Psi).
        As a consequence we need not build the full dense H, but only 
        its two halves separately, since the Kronecer product is associative.

        Here the 'vec' operator stacks each column of a matrix into a single
        column vector. Since Python is row-major we can implement this as
        vec(A) = np.ravel(A.T).
        """
        self.__storehalves()
        Psimat = psi.reshape((self.leftT.shape[0], self.right.shape[1])).T
        ans = vec(reduce(np.dot, [self.right, Psimat, self.leftT]))
        return ans

    def todense(self, start=0, end=-1):
        if end==-1:
            end = self.N
        return reduce(np.kron, self.chain[start:end])

    def timesvectordense(self, psi):
        """
        This function is identical to chaintimesvector except that it does not
        exploit the Kronecker identity used by the latter. This is useful for
        testing.
        """
        matrix = self.todense()
        if matrix.shape[1] != psi.shape[0]:
            raise IndexError("Invalid dimensions; H: ", matrix.shape, "psi: ", 
                             psi.shape)
        return np.dot(matrix, psi)


def nnchain2d(sitex, sitey, Nx, Ny, siteterm, nextterm, coef=1, d=2):
    """
    Return the two OperatorChains corresponding to the nearest neighbour
    couplings in 
    H = Sum_<ij> coef * (siteterm_i \otimes nextterm_j) for a particular
    value of i, assuming PBCs. The first (second) chain couples the ith site
    to its immediate right (bottom). "None" is returned if this would couple
    a site to itself (i.e. in the 1D case).
    """
    if sitex >= Nx or sitey >= Ny:
        raise IndexError("Invalid index; sitex, Nx, sitey, Ny:", 
                            sitex, Nx, sitey, Ny)
    if siteterm.shape != (d, d) or nextterm.shape != (d, d):
        raise ValueError("Bad term shapes ", siteterm.shape, nextterm.shape)
    N = Nx*Ny
    thisidx = sitex + Nx * sitey
    ones = [np.eye(d, dtype=np.complex128)]*N
    ones[thisidx] = coef*siteterm

    rightx = (sitex+1)%Nx
    if rightx == sitex:
        rightchain = None
    else:
        rights = copy(ones)
        rightidx = rightx + Nx*sitey
        rights[rightidx] = nextterm
        #print "rights: ", [np.abs(r) for r in rights]
        rightchain = OperatorChain(rights)

    downy = (sitey+1)%Ny
    if downy == sitey:
        downchain = None
    else:
        downs = copy(ones)
        downidx = sitex + Nx*downy
        downs[downidx] = nextterm
        #print "downs: ", [np.abs(d) for d in downs]
        downchain = OperatorChain(downs)

    return rightchain, downchain

#*****************************************************************************
#Functions for testing
#*****************************************************************************
def frobnorm(A, B):
    return npla.norm(np.ravel(A) - np.ravel(B))

def random_rng(shp, low=-1, high=1):
    return (high - low) * np.random.random_sample(shp) + low

def random_complex(shp, real_low=-1.0, real_high=1.0, imag_low=-1.0, 
        imag_high=1.0):
    """Return a normalized randomized complex matrix of shape shp.
    """
    realpart = random_rng(shp, low=real_low, high=real_high)
    imagpart = 1.0j * random_rng(shp, low=imag_low, high=imag_high)
    bare = realpart + imagpart
    #bare /= la.norm(bare)
    return bare

def random_hermitian(diaglength):
    A = random_complex((diaglength, diaglength))
    return 0.5*(A+np.conj(A).T)

def random_unitary(diaglength):
    A = random_hermitian(diaglength)
    return spla.expm(-1.j*A)



def nnterm(n, N, left, right, c=1):
    """
    Construct a length-N dense term of the form c*left \otimes right, 
    where left acts at site n and right at site n+1. In other words
    this returns c* I \otimes I \otimes ... left \otimes right \otimes I...

    If right is None, it is set to the identity, and so this constructs a local
    term.

    If right is -1, left is assumed to already be a nearest-neighbour term.
    """
    if n>N:
        raise IndexError("n= ", n, "too large for N=", N)
    d = left.shape[0]
    if left.shape != (d, d):
        raise ValueError("left had weird shape", left.shape)
    if right.shape != (d, d):
        raise ValueError("right had weird shape", right.shape)
    one = np.eye(d)
    
    chain = [one]*N
    chain[n] = left
    chain[(n+1)%N] = right
    #print "chain: ", [np.abs(ch) for ch in chain]
    term = reduce(np.kron, chain)

    return c*term


def nnhamiltonian(N, left, right, coef=1):
    """
    Construct a length-N dense Hamiltonian 
    """
    term = 0
    for n in range(0, N):
        term += nnterm(n, N, left, right, c=coef) 
    return term

# def makeisingham(J, h, N):
    # """
    # The length-N dense transverse Ising Hamiltonian.
    # Sum_i (J*XX + h*ZI) 
    # """
    # term = - (J*np.kron(Sig_x, Sig_x) + (h/2.0) * np.kron(Sig_z, np.eye(2)) + 
              # (h/2.0) * np.kron(np.eye(2), Sig_z))
    # H = nnhamiltonian(N, term)
    # return H

def checkfunction(f, thresh, Ntimes):
    err = sum(itertools.repeat(f(), Ntimes))/Ntimes
    print "err: ", err
    if err < thresh:
        print "Passed!"
    else:
        print "Failed!"

#*****************************************************************************
#TEST 1
#*****************************************************************************
def testchaintimesvector(N, d=2, thresh=1E-10, Ntimes=1):
    """
    Make sure the dense and sparse versions of chaintimesvector give the
    same answer.
    """
    print "Sparse vs dense OperatorChains: " 
    def sparsedense():
        chain = [random_complex((D,D)) for D in itertools.repeat(d, N)]
        opchain = OperatorChain(chain)
        vec = random_complex((d**N))
        dense = opchain.timesvectordense(vec)
        sparse = opchain.timesvector(vec)
        return frobnorm(dense, sparse)
    checkfunction(sparsedense, thresh, Ntimes)

    print "Dense OperatorChain vs np.array: " 
    def densearray():
        chain = [random_complex((D,D)) for D in itertools.repeat(d, N)]
        vec = random_complex((d**N))
        opchain = OperatorChain(chain)
        dense = opchain.timesvectordense(vec)
        array = reduce(np.kron, chain)
        timesarray = np.dot(array, vec)
        return frobnorm(dense, timesarray)
    checkfunction(densearray, thresh, Ntimes)

#*****************************************************************************
#TEST 2
#*****************************************************************************
def testdensehamiltonianconstruction(N, d=2, thresh=1E-10, Ntimes=1, Nops=1):
    def checkdensebuild(x=True):
        coefs = random_complex((Nops))
        leftops = [random_hermitian(D) for D in itertools.repeat(d, Nops)] 
        rightops = [random_hermitian(D) for D in itertools.repeat(d, Nops)] 

        denseham = 0.0
        for coef, left, right in itertools.izip(coefs, leftops, rightops):
            denseham += nnhamiltonian(N, left, right, coef)

        if x:
            classham = nnhamfactory(1, N, leftops, rightops, coefs, d=d, 
                                    sparse=False)
        else:
            classham = nnhamfactory(N, 1, leftops, rightops, coefs, d=d,
                                    sparse=False)

        return frobnorm(denseham, classham.data)
    print "Nx=1 : "
    checkfunction(lambda : checkdensebuild(x=False), thresh, Ntimes)

    print "Ny=1 :"
    checkfunction(lambda : checkdensebuild(x=True), thresh, Ntimes)

    print "Nx vs Ny: "
    def checkNxNy():
        coefs = random_complex((Nops))
        leftops = [random_hermitian(D) for D in itertools.repeat(d, Nops)] 
        rightops = [random_hermitian(D) for D in itertools.repeat(d, Nops)] 

        classhamx1 = nnhamfactory(1, N, leftops, rightops, coefs, d=d, 
                                    sparse=False)
        classhamy1 = nnhamfactory(N, 1, leftops, rightops, coefs, d=d,
                                    sparse=False)
        return frobnorm(classhamx1.data, classhamy1.data)
    checkfunction(checkNxNy, thresh, Ntimes)
#*****************************************************************************
#TEST 3
#*****************************************************************************
def checksparsedense(Nx, Ny, d=2, Ntimes=5, Nops=3):
    errham = 0.0
    errmatvec = 0.0
    erreigvals = 0.0
    for dummy in range(0, Ntimes):
        coefs = random_rng((Nops))
        leftops = [random_hermitian(D) for D in itertools.repeat(d, Nops)] 
        rightops = [random_hermitian(D) for D in itertools.repeat(d, Nops)] 
        
        denseham = nnhamfactory(Nx, Ny, leftops, rightops, coefs, d=d, 
                                sparse=False)
        sparseham = nnhamfactory(Nx, Ny, leftops, rightops, coefs, d=d, 
                                sparse=True)

        sparsedense = sparseham.densedata()
        
        vec = random_complex((d**(Nx*Ny)))
        vecdense = denseham.matvec(vec)
        vecsparse = sparseham.matvec(vec)
        
        evs, eVs = denseham.eigs()
        evsparse, eVsparse = sparseham.eigs(k=2)
        
        errham += frobnorm(denseham.data, sparsedense)
        errmatvec += frobnorm(vecdense, vecsparse)
        erreigvals += frobnorm(evs[:2], evsparse)

    print "err(H): ", errham/Ntimes
    print "err(H*vec): ", errmatvec/Ntimes
    print "err(eigvals): ", erreigvals/Ntimes


#*****************************************************************************
#TEST 4
#*****************************************************************************
def pairterm(A, B, c, pair, N, d):
    term = [np.eye(d)]*N
    term[pair[0]] = A
    term[pair[1]] = B
    return c*reduce(np.kron, term)

def buildtwobytwo(A, B, c, d=2):
    pairs = [(0,1),(0,2),(1,0),(1,3),(2,3),(2,0),(3,1),(3,2)]
    term = 0.0
    for pair in pairs:
        term += pairterm(A, B, c, pair, 4, d)
    return term

def buildthreebythree(A, B, c, d=2):
    pairs = [(0,1),(0,3),(1,2),(1,4),(2,0),(2,5),(3,4),(3,6),(4,5),
             (4,7),(5,3),(5,8),(6,7),(6,0),(7,8),(7,1),(8,6),(8,2)]
    term = 0.0
    for pair in pairs:
        term += pairterm(A, B, c, pair, 9, d)
    return term

def checkhandbuilt(d=2, Ntimes=1, Nops=1):
    for N in [2, 3]:
        errham = 0.0
        errmatvec = 0.0
        erreigvals = 0.0
        for _ in range(0, Ntimes):
            coefs = random_rng((Nops))
            leftops = [random_hermitian(D) for D in itertools.repeat(d, Nops)] 
            rightops = [random_hermitian(D) for D in itertools.repeat(d, Nops)] 
            
            denseham = nnhamfactory(N, N, leftops, rightops, coefs, d=d, 
                                    sparse=False)
            explicit = 0.0
            for coef, left, right in itertools.izip(coefs, leftops, rightops):
                if N==2:
                    explicit += buildtwobytwo(left, right, coef, d=d)
                elif N==3:
                    explicit += buildthreebythree(left, right, coef, d=d)
                else:
                    raise ValueError("How did we get here?")

            vec = random_complex((d**(N*N)))
            vecdense = denseham.matvec(vec)
            vecexplicit = np.dot(explicit, vec)

            evs, eVs = denseham.eigs()
            evsexp, eVsexp = npla.eigh(explicit)
                
            errham += frobnorm(denseham.data, explicit)
            errmatvec += frobnorm(vecdense, vecexplicit)
            erreigvals += frobnorm(evsexp, evs)

                                
        print "N=", N
        print "err(H): ", errham
        print "err(H*vec): ", errmatvec
        print "err(eigvals): ", erreigvals
            
            

#*****************************************************************************
#TEST 5
#*****************************************************************************

def checkIsing(Nx, Ny, Js=(0., 0.5, 1.0, 2.0), 
              hs=(0., 0.5, 1.0, 2.0), Ntimes=10):
    
    errham=0.0
    errmatvec=0.0
    erreigvals=0.0
    for J, h in itertools.product(Js, hs):
        if J+h != 0:
            for dummy in range(0, Ntimes):
               denseham = makeisingham(Nx, Ny, J, h, sparse=False)  
               sparseham = makeisingham(Nx, Ny, J, h, sparse=True)  
            
               sparsedense = sparseham.densedata()
                
               vec = random_complex((2**(Nx*Ny)))
               vecdense = denseham.matvec(vec)
               vecsparse = sparseham.matvec(vec)
                
               evs, eVs = denseham.eigs()
               evsparse, eVsparse = sparseham.eigs(k=1)
                
               errham += frobnorm(denseham.data, sparsedense)
               errmatvec += frobnorm(vecdense, vecsparse)
               # print evs[:4]
               # print evsparse
               erreigvals += frobnorm(evs[:1], evsparse)

    print "err(H): ", errham/Ntimes
    print "err(H*vec): ", errmatvec/Ntimes
    print "err(eigvals): ", erreigvals/Ntimes

def IsingTest(Nx, Ny):
    sparseham = makeisingham(Nx, Ny, 2.0, 0.0, sparse=True)  
    evsparse, eVsparse = sparseham.eigs(k=1)
    if Nx==1 or Ny==1:
        correct = -2.0
    else:
        correct = -4.0

    print "err (h=0): ", np.sqrt((correct - evsparse/(Nx*Ny))**2)

    sparseham = makeisingham(Nx, Ny, 0.0, 2.0, sparse=True)
    evsparse, eVsparse = sparseham.eigs(k=1)
    print "err (J=0): ", np.sqrt((correct-evsparse/(Nx*Ny))**2)


    
        



            
        






