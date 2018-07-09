import numpy as np
from copy import copy, deepcopy
import numpy.linalg as npla
from scipy import linalg as spla
import scipy.sparse.linalg
import itertools

#******************************************************************************
#Clifford gates
#******************************************************************************
Sig_x = np.array([ [0+0.j, 1+0.j],
                   [1+0.j, 0+0.j]] )

Sig_y = np.array([ [0, -1.j],
                   [1.j, 0]] )

Sig_z = np.array([ [1,  0+0.j],
                   [0, -1]] )

hadamard_gate = (1./np.sqrt(2))*np.array( [ 1.+0.j, 1.+0.j],
                                          [1.+0.j, -1.+0.j] )

phase_gate =  np.array( [0.+0.j, 0.+0.j],
                        [0.+0.j, 0.+1.j] )

HSdag = np.dot(hadamard_gate, np.conj(phase_gate.T))

basisdict = { 'X' : hadamard_gate,
              'Y' : HSdag,
              'Z' : None
            }
#*****************************************************************************
#ModelWavefunctions
#*****************************************************************************
class ModelWavefunction(object):
    def __init__(self, Nx, Ny, lefts, rights, coefs, state=0, sparse=False, d=2):
        self.Nx = Nx
        self.Ny = Ny
        self.d=d
        self.sparse=sparse
        self.state=state
        self.ham = nnhamfactory(Nx, Ny, lefts, rights, coefs, d=2, sparse=sparse)
        w, v = self.ham.eigs(k=state+1)
        self.E = w[state]
        self.wf = Wavefunction(self.E, v[:, state])
    
def optochain(n, op):
    if op.shape != (self.d, self.d):
        raise ValueError("op had bad shape ", op.shape)
    chain = np.eye(self.d)*self.N
    chain[n] = op
    opchain = OperatorChain(chain)
    return opchain



class IsingWavefunction(ModelWavefunction):
    def __init__(self, Nx, Ny, J, h, state=0, sparse=False, d=2):
        self.J = J
        self.h = h
        lefts = [Sig_x, np.eye(2), Sig_z]
        rights = [Sig_x, Sig_z, np.eye(2)]
        coefs = [-J, -h/2.0, -h/2.0]
        super(IsingWavefunction, self).__init__(Nx, Ny, lefts, rights, coefs,
                                                state=state, sparse=sparse, d=d)
class Wavefunction(object):
    """Represents a wavefunction on a Hilbert space
       H = H1^d \otimes H2^d \otimes H3^d ... H^Nd.
       The coefficients are stored in self.wf which has shape self.shape.
    """
    def __init__(self, wf, d=2, N=N):
        self.shape = wf.shape
        self.wf = wf
        self.d = d
        self.N = N
    
    def cascadedmeasurement(self, basis, outcome=collections.deque()):
        #thisgate = gatedict(basis.pop())
        #rotatedwf = self.applygate(thisgate, self.N)
        thisoutcome, projectedwf = rotatedwf.measure(rotatedwf.N, basis.pop())
        outcome.appendleft(thisoutcome)
        if projectedwf.N == 0:
            return list(outcome)
        else:
            return projectedwf.cascadedmeasurement(basis, outcome=outcome)

    def measure(self, site, basis='Z'):
        """
        Simulates a measurement upon the 'site'th qbit in the Pauli basis
        'basis', which is the computational (Z) basis by default. An error
        is thrown unless basis is either 'X', 'Y', or 'Z'. 

        computational
        basis.
        """
        try:
            thisgate = gatedict(basis)
        except KeyError:
            raise ValueError("Invalid basis choice ", basis)


        rotatedwf = self.applygate(thisgate, self.N)
        outcome = 1
        projectedwf = self.wf
        
        return outcome, projectedwf

    def expectationvaluesite(self, n, op):
        """
        <psi| I1 \otimes I2 \otimes...op_n... I_N |psi>
        """
        opchain = optochain(n, op)
        return self.expectationvaluechain(opchain)

    def expectationvaluechain(self, opchain):
        """
        <psi| Op1 \otimes Op2 \otimes... Op_n |psi>
        """
        return np.vdot(self.wf, self.applygatechain(opchain))

    def applygate(self, n, op):
        """
        |psi'> = I_1 \otimes I_2 ... op_n ... I_N |psi>
        """
        if op is None:
            return Wavefunction(self.wf, d=self.d, N=self.N)
        opchain = optochain(n, op)
        return applygatechain(opchain)

    def applygatechain(self, opchain):
        """
        |psi'> = Op1 \otimes Op2 \otimes... Opn |psi>
        The ops are an OperatorChain. 
        """

        opvec = opchain.timesvec(self.wf)
        return Wavefunction(opvec, d=self.d, N=self.N)

    def norm(self):
        """
        <\psi|\psi>
        """
        return self.overlap(self.wf)

    def overlap(self, other):
        """
        <\phi | \psi>
        """
        return np.vdot(other.wf, self.wf)



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
        chit = itertools.izip(leftterms, rightterms, coefs)
        grid = itertools.product(range(0, Ny), range(0, Nx))
        for left, right, coef in chit:
            for y, x, in grid:
                self.data += self.maketerm(x, y, left, right) 
            self.data *= coef

    def maketerm(sitex, sitey, siteterm, nextterm):
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
            rightterm = rights.todense()
        if downchain is not None:
            downterm = downs.todense()
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

    def eigs(self, k):
        """The first k eigenvalues and normalized eigenvectors.
        """
        w, v = npla.eigh(self.data)
        w = w[:k]
        v = v[:, :k]
        #v /= npla.norm(v, axis=0)
        return w, v
    

class NNHamiltonianSparse(NNHamiltonian):
    def __init__(self, Nx, Ny, leftterms, rightterms, coefs, d=2):
        """Build from three lists: leftterms, rightterms, and coefs. 
           Each of these represents a term of the form
           coefs[n] * Sum_<ij> (leftterms[n]_i rightterms[n]_j). 
        """
        super(NNHamiltonianSparse, self).__init__(Nx, Ny, leftterms, 
                                                  rightterms, coefs, d=d)
        chit = itertools.izip(leftterms, rightterms, coefs)
        grid = itertools.product(range(0, Ny), range(0, Nx))
        for left, right, coef, in chit:
            for y, x, in grid:
                self.chains = self.chains + self.maketerm(x, y, left, right, coef) 


    def maketerm(sitex, sitey, siteterm, nextterm, coef):
        grid = itertools.product(range(0, Ny), range(0, Nx))
        term = []
        for y, x in grid:
            right, down = nnchain2d(x, y, self.Nx, self.Ny, siteterm, nextterm,
                                    coef=coef, d=self.d)
            if right is not None:
                term = term + [right,]
            if down is not None:
                term = term + [down,]
        return term

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
        return output

    def aslinearoperator(self, shift=0):
        shiftmatvec = lambda(vec): self.matvec(vec, shift=shift)
        return scipy.sparse.linalg.LinearOperator(self.shape, shiftmatvec)

    def eigs(self, k=1, shift="auto"):
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
            shift = 100*self.Nx * self.Ny
        op = self.aslinearoperator(shift=shift)
        w, v = sp.sparse.linalg.eigsh(op, k=k)
        w = w + shift
        v /= npla.norm(v, axis=0)
        return w, v

def nnhamfactory(Nx, Ny, lefts, rights, coefs, d=2, sparse=False):
    if sparse:
        return NNHamiltonianSparse(Nx, Ny, lefts, rights, coefs, d=2)
    else:
        return NNHamiltonianDense(Nx, Ny, lefts, rights, coefs, d=2)

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
        self.d = chain[0].shape[0]
        if not all([link.shape==(self.d,self.d) for link in chain]):
            raise ValueError("chain had bad shape: ", [link.shape for link in chain])
        self.halvesready = False

    def __storehalves(self):
        if not self.halvesready:
            halfN = int(round(N/2))
            BT = self.todense(end=halfN).T
            A = self.todense(start=halfN)
            self.leftT = BT
            slef.right = A
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
        Psimat = psi.reshape((self.left.shape[0], self.right.shape[1])).T
        ans = vec(reduce(np.dot, [self.right, Psimat, self.left]))
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
        if end==-1:
            end=self.N
        matrix = self.todense(start=start, end=end)
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
    ones = [coef*np.eye(d, dtype=np.complex128)]*N
    ones[thisidx] = coef*siteterm

    rightx = (sitex+1)%Nx
    if rightx == sitex:
        rights = None
    else:
        rights = copy(ones)
        rightidx = rightx + Nx*sitey
        rights[rightidx] = coef*nextterm

    downy = (sitey+1)%Ny
    if downy == sitey:
        downs = None
    else:
        downs = copy(ones)
        downidx = sitex + Nx*downy
        downs[downidx] = coef*nextterm

    rightchain = OperatorChain(rights)
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

def testchaintimesvector(N, d=2, thresh=1E-12, Ntimes=1):
    """
    Make sure the dense and sparse versions of chaintimesvector give the
    same answer.
    """
    err=0.
    for thistime in range(0, Ntimes):
        chain = []
        for i in range(0, N):
            chain.append(random_complex((d,d)))
        vec = random_complex((d**N))
        dense = chaintimesvectordense(chain, vec)
        sparse = chaintimesvectorsparse(chain, vec)
        err = 0
    if err < thresh:
        print "Passed!"
    else:
        print "Failed!"
    








