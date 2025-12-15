using Test: @test, @testset
using ITensors
using ITensorNetworks: dmrg, maxlinkdim, siteinds, ttn
using Graphs: dst, edges, src, vertices
using ITensorMPS: OpSum
using TensorOperations: TensorOperations #For contraction order finding
using Suppressor: @capture_out

include("utilities/simple_ed_methods.jl")
include("utilities/tree_graphs.jl")

@testset "Tree DMRG" begin
    outputlevel = 0

    g = build_tree(; nbranch = 3, nbranch_sites = 3)

    sites = siteinds("S=1/2", g)

    # Make Heisenberg model Hamiltonian
    h = OpSum()
    for edge in edges(sites)
        i, j = src(edge), dst(edge)
        h += "Sz", i, "Sz", j
        h += 1 / 2, "S+", i, "S-", j
        h += 1 / 2, "S-", i, "S+", j
    end
    H = ttn(h, sites)

    # Make initial product state
    state = Dict{Tuple{Int, Int}, String}()
    for (j, v) in enumerate(vertices(sites))
        state[v] = iseven(j) ? "Up" : "Dn"
    end
    psi0 = ttn(state, sites)

    (outputlevel >= 1) && println("Computing exact ground state")
    Ex, psix = ed_ground_state(H, psi0)
    (outputlevel >= 1) && println("Ex = ", Ex)

    cutoff = 1.0e-5
    maxdim = 40

    factorize_kwargs = (; cutoff, maxdim)

    nsweeps = 5

    #
    # Test 2-site DMRG without subspace expansion
    #
    nsites = 2
    E, psi = dmrg(H, psi0; factorize_kwargs, nsites, nsweeps, outputlevel)
    (outputlevel >= 1) && println("2-site DMRG energy = ", E)
    @test E ≈ Ex atol = 1.0e-5

    #
    # Test 1-site DMRG with subspace expansion
    # and cutoff and maxdim as vectors of values
    #
    nsites = 1
    nsweeps = 5
    maxdim = [8, 16, 32]
    factorize_kwargs = (; cutoff = [1.0e-5, 1.0e-6], maxdim)
    extract!_kwargs = (; subspace_algorithm = "densitymatrix")
    E, psi = dmrg(H, psi0; extract!_kwargs, factorize_kwargs, nsites, nsweeps, outputlevel)
    (outputlevel >= 1) && println("1-site+subspace DMRG energy = ", E)
    @test E ≈ Ex atol = 1.0e-5

    # Regression test that subspace expansion feature obeys maxdim limit
    @test maxlinkdim(psi) <= last(maxdim)

    #
    # Test passing cutoff and maxdim as a vector of values
    #
    nsites = 2
    factorize_kwargs = (; cutoff = [1.0e-5, 1.0e-6], maxdim = [8, 16, 32])
    E, psi = dmrg(H, psi0; factorize_kwargs, nsites, nsweeps, outputlevel = 0)
    (outputlevel >= 1) && println("2-site DMRG energy = ", E)
    @test E ≈ Ex atol = 1.0e-5

    #
    # Test that outputlevel > 0 generates output
    # and outputlevel == 0 generates no output
    #
    nsweeps = 2
    outputlevel = 1
    output = @capture_out begin
        dmrg(H, psi0; factorize_kwargs, nsweeps, outputlevel)
    end
    @test length(output) > 0

    outputlevel = 0
    output = @capture_out begin
        dmrg(H, psi0; factorize_kwargs, nsweeps, outputlevel)
    end
    @test length(output) == 0
end
