

using ArgParse, LinearMaps, HDF5, JLD


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
      "--hfield" 
        help = "Magnetic field parameter."
        arg_type = Real
        default = 2.0
      "--Jcoupling", "-J"
        help = "Coupling constant."
        arg_type = Real
        default = 1.0
      "--Nx", "-x"
        help = "Number of columns in the lattice."
        arg_type = Int
        default = 10
      "--Ny", "-y"
        help = "Number of rows in the lattice."
        arg_type = Int
        default = 1
      "--outputpath", "-o"
        help = "Where to save output. Default is ./ising. The full output 
                string is formatted to also include the arguments; e.g.
                ./ising_h1_0_J1_0Nx3Ny3.out"
        arg_type = AbstractString
        default =  "./ising"
    end

    return parse_args(s)
end

"""
This makes the nearest-neighbour Hamiltonian, which is a 4x4 matrix.
"""
function buildisingham(J, h)
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    I2 = eye(2)
    XX = kron(X, X)
    ZI = kron(Z, I2)
    IZ = kron(I2, Z)
    H = -(J*XX + h/2*(ZI+IZ))
    return H
end

"""
This is the function that actually multiplies the input by the hamiltonian.
It does this once for each cycle in the permutation; the length of the cycle
needs to be supplied..
"""
function applycycleperm(v, out, ham, cycleperm, cyclelength)
    D = length(v)
    N = convert(Int64, log2(D))
    siteshp = tuple(collect(Iterators.repeated(2,N))...)#siteshp=(2,2,2..) i.e. each dim is a site
                                              #  with one entry per spin state
    for n=1:cyclelength
        #reshape so we can apply ham
        v, out = reshape(v, (4, 2^(N-2))), reshape(out, (4,2^(N-2)))
        out += ham*v
        #reshape so we can permute sites (since each gets its own dim)
        v, out = reshape(v, siteshp), reshape(out, siteshp) 
        v, out = permutedims(v, cycleperm), permutedims(out, cycleperm)
    end
    out = reshape(out, D)
    return v, out
end

"""
Constructs a cyclic permutation tuple of length N, which is a 1:N range
whose entries at the indices permuteidx are replaced by the values in 
permutedval. For example, N=5, permuteidx=(1,3,5), permutedval=(5,1,3)
would return cycleperm=(5,2,1,4,3).
"""
function makecycleperm(permuteidx, permutedval, N)
    L = length(permuteidx)
    @assert L==length(permutedval) "$L, $length(permutedval)"
    cyclepermarr = collect(1:N)
    for p =  1:L
      idx = permuteidx[p]
      cyclepermarr[idx] = permutedval[p]
    end
    cycleperm = tuple(cyclepermarr...)
    return cycleperm
end

function addperm!(first::Int, stride::Int, last::Int, permuteidxs, permutedvals) 
      theseidxs = collect(first:stride:last) #e.g. [4,5,6] if 2nd col of 3x3
      thesevals = circshift(theseidxs, -1) #e.g. [5,6,4]

      push!(permuteidxs, theseidxs)
      push!(permutedvals, thesevals)
      # #need to also push a cyclic permutation to cover the PBCs
      # push!(permuteidxs, circshift(theseidxs, 1))
      # push!(permutedvals, circshift(thesevals, 1))
end


"""
This function applies the full Hamiltonian to the input state v, by 
repeatedly applying the nearest-neighbour Hamiltonian to appropriate 
permutations of v such that each nearest-neighbour pair is coupled. 
"""
function applyspinham(v::AbstractVector, ham::Matrix, Nx::Int, Ny::Int)
    D = length(v)
    N = Nx*Ny
    @assert D==2^N "D ($D) didn't equal 2^N (2^$N)"
    siteshp = tuple(collect(Iterators.repeated(2,N))...)#siteshp=(2,2,2..) i.e. each dim is a site
                                              #  with one entry per spin state
    out = zeros(siteshp)

    #can probably pre-allocate these, but they are gonna be short enough that
    #little would be gained
    permuteidxs = []
    permutedvals = []

    #It's probably possible to subsume these two for loops into a single
    #more general function, but meh.
    #Make the column permutations
    if Ny>1
      for col = 1:Nx
        first = (col-1)*Ny+1
        last = (col-1)*Ny+Ny
        addperm!(first, 1, last, permuteidxs, permutedvals)
      end
    end 
    
    #Make the row permutations
    if Nx>1
      for row = 1:Ny
        first = row
        last = (Nx-1)*Ny+row
        addperm!(first, Ny, last, permuteidxs, permutedvals)
      end
    end
    # println("permuteidxs:")
    # println(permuteidxs)
    # println("permutedvals:")
    # println(permutedvals)
     
    #Apply the Hamiltonian once for each permutation
    for p in zip(permuteidxs,  permutedvals)
      #println(p[1], p[2])
      cycleperm = makecycleperm(p[1], p[2], N)
      v, out = applycycleperm(v, out, ham, cycleperm, length(p[1]))
    end
    out = reshape(out, D)
    return out
end


"""
This wraps all the various calls needed to find the Ising ground state because
I don't like long main functions. The Julia function eigs() implements 
the Lancoz eigensolver, which conveniently requires the target operator to
be defined only through its action on vectors (i.e. Julia's LinearMap type).
"""
function findgroundstate_ising(parsed_args::Dict)
  Nx = parsed_args["Nx"]
  Ny = parsed_args["Ny"]
  N = Nx*Ny
  # rng = MersenneTwister()
  # v = randn(rng, 2^N) + randn(rng, 2^N)*im
  isingham = buildisingham(parsed_args["Jcoupling"],  parsed_args["hfield"]) 
  applyisingham(v) = applyspinham(v, isingham, Nx, Ny)
  H = LinearMap(applyisingham, 2^N)
  u, v = eigs(H; nev=1, which=:SR)
  v/=sqrt(sum(abs2(v))) #normalize
  energy = real(u[1])#/N
  edensity = energy/N
  #infN_e_density = -4/pi
  println("Energy:", energy)
  println("Energy density:", edensity)
  #info((edensity - infN_e_density)/infN_e_density)
  return energy, v 
end

"""
Little function to save the output to disk.
"""
function saveoutput(parsed_args::Dict, energy::Real, 
                    wavefunction::AbstractArray) 
    outbuffer = IOBuffer()
    print(outbuffer, parsed_args["outputpath"])
    print(outbuffer, string("_h", 
                            replace(string(parsed_args["hfield"]), ".", "_")))
    print(outbuffer, string("J", 
                            replace(string(parsed_args["Jcoupling"]), ".",  "_")))
    print(outbuffer, string("Nx", parsed_args["Nx"]))
    print(outbuffer, string("Ny", parsed_args["Ny"]))
    print(outbuffer, ".jld")
    outstring = takebuf_string(outbuffer)
    println("Saving to ", outstring)
    save(outstring, "energy", energy)
    save(outstring, "wavefunction", wavefunction)
end

function main()
  parsed_args = parse_commandline()
  println("Hi!! Finding the Ising model ground state with the following arguments:")
  for (arg, val) in parsed_args
    println("   $arg    =>    $val")
  end
  energy, wavefunction = findgroundstate_ising(parsed_args)  
  saveoutput(parsed_args, energy, wavefunction)
end
main()
