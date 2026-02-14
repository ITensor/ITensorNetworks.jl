@eval module $(gensym())
using Graphs: vertices
using ITensorNetworks.ITensorsExtensions: group_terms
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using ITensorNetworks: ITensorNetwork, cartesian_to_linear, dmrg, expect, siteinds, tebd
using ITensors: ITensors
using NamedGraphs.GraphsExtensions: rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @test_broken, @testset

ITensors.disable_warn_order()

@testset "Ising TEBD" begin
    dims = (2, 3)
    n = prod(dims)
    g = named_grid(dims)

    h = 0.1

    s = siteinds("S=1/2", g)

    #
    # PEPS TEBD optimization
    #
    ℋ = ModelHamiltonians.ising(g; h)
    χ = 2
    β = 2.0
    Δβ = 0.2

    ψ_init = ITensorNetwork(v -> "↑", s)
    #E0 = expect(ℋ, ψ_init)
    ψ = tebd(
        group_terms(ℋ, g),
        ψ_init;
        β,
        Δβ,
        cutoff = 1.0e-8,
        maxdim = χ,
        ortho = false,
        print_frequency = typemax(Int)
    )
    #E1 = expect(ℋ, ψ)
    ψ = tebd(
        group_terms(ℋ, g),
        ψ_init;
        β,
        Δβ,
        cutoff = 1.0e-8,
        maxdim = χ,
        ortho = true,
        print_frequency = typemax(Int)
    )
    #E2 = expect(ℋ, ψ)
    #@show E0, E1, E2, E_dmrg
    @test_broken (((abs((E2 - E1) / E2) < 1.0e-3) && (E1 < E0)) || (E2 < E1 < E0))
end
end
