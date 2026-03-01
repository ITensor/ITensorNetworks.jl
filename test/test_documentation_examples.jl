@eval module $(gensym())
using ITensorNetworks: ITensorNetworks, ITensorNetwork, IndsNetwork, inner, mps, norm,
    random_mps, random_ttn, siteinds, ttn
using ITensors
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using TensorOperations
using Test

@testset "ITensorNetworks" begin

    # 3×3 square-lattice tensor network
    g = named_grid((3, 3))
    s = siteinds("S=1/2", g)            # one spin-½ Index per vertex

    # Zero-initialized, bond dimension 2
    ψ = ITensorNetwork(s; link_space = 2)

    # Product state — every site in the |↑⟩ state
    ψ = ITensorNetwork("Up", s)

    # Staggered initialization with a vertex-dependent function
    ψ = ITensorNetwork(v -> isodd(sum(v)) ? "Up" : "Dn", s)

    i, j, k = Index(2, "i"), Index(2, "j"), Index(2, "k")
    A, B, C = ITensor(i, j), ITensor(j, k), ITensor(k)
    tn = ITensorNetwork([A, B, C])                     # integer vertices 1, 2, 3
    tn = ITensorNetwork(["A", "B", "C"], [A, B, C])       # named vertices
    tn = ITensorNetwork(["A" => A, "B" => B, "C" => C])       # from pairs
end

@testset "Tree Tensor Networks" begin

    # Comb-tree TTN (a popular tree topology for 2D-like systems)
    g = named_comb_tree((4, 3))
    sites = siteinds("S=1/2", g)

    psi = ttn(sites)              # zero-initialised
    psi = ttn(v -> "Up", sites)   # product state

    # Random, normalised TTN
    psi = random_ttn(sites; link_space = 4)

    # 1D MPS
    s1d = siteinds("S=1/2", 10)
    mps_state = mps(v -> "Up", s1d)   # product MPS
    mps_state = random_mps(s1d; link_space = 4)

    # Comb-tree TTN with random bond-dimension-2 tensors
    g = named_comb_tree((3, 4))
    s = siteinds("S=1/2", g)
    psi = ttn(v -> "Up", s)

    # Convert from a dense ITensor
    g = named_comb_tree((3, 1))
    sites = siteinds("S=1/2", g)
    A = ITensors.random_itensor(sites[(1, 1)], sites[(2, 1)], sites[(3, 1)])
    ttn_A = ttn(A, sites)
end

@testset "Computing Properties" begin
    g = named_comb_tree((4, 3))
    sites = siteinds("S=1/2", g)

    phi = random_ttn(sites; link_space = 4)
    psi = random_ttn(sites; link_space = 4)
    z = inner(phi, psi)               # ⟨ϕ|ψ⟩  (belief propagation by default)
    n = norm(psi)                     # √⟨ψ|ψ⟩

    phi = ITensorNetwork(random_ttn(sites; link_space = 4))
    psi = ITensorNetwork(random_ttn(sites; link_space = 4))
    z = inner(phi, psi)               # ⟨ϕ|ψ⟩  (belief propagation by default)
    z = inner(phi, psi; alg = "exact")  # ⟨ϕ|ψ⟩  (exact contraction)
    n = norm(psi)                     # √⟨ψ|ψ⟩
end

end
