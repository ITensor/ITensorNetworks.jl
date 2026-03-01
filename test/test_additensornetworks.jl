@eval module $(gensym())
using Graphs: rem_edge!, vertices
using ITensorNetworks: ITensorNetwork, inner_network, orthogonalize, random_tensornetwork,
    siteinds, truncate, ttn
using ITensors: ITensors, apply, inner, op, scalar
using LinearAlgebra: norm_sqr
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NamedGraphs: NamedEdge
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @testset
@testset "add_itensornetworks" begin
    g = named_grid((2, 2))
    s = siteinds("S=1/2", g)
    ψ1 = ITensorNetwork(v -> "↑", s)
    ψ2 = ITensorNetwork(v -> "↓", s)

    ψ_GHZ = ψ1 + ψ2

    v = (2, 2)
    Oψ_GHZ = copy(ψ_GHZ)
    Oψ_GHZ[v] = apply(op("Sz", s[v]), Oψ_GHZ[v])

    ψψ_GHZ = inner_network(ψ_GHZ, ψ_GHZ)
    ψOψ_GHZ = inner_network(ψ_GHZ, Oψ_GHZ)

    @test scalar(ψOψ_GHZ) / scalar(ψψ_GHZ) == 0.0

    χ = 3
    s1 = siteinds("S=1/2", g)
    s2 = copy(s1)
    rem_edge!(s2, NamedEdge((1, 1) => (1, 2)))

    v = rand(vertices(g))
    rng = StableRNG(1234)
    ψ1 = random_tensornetwork(rng, s1; link_space = χ)
    ψ2 = random_tensornetwork(rng, s2; link_space = χ)

    ψ12 = ψ1 + ψ2

    Oψ12 = copy(ψ12)
    Oψ12[v] = apply(op("Sz", s1[v]), Oψ12[v])

    Oψ1 = copy(ψ1)
    Oψ1[v] = apply(op("Sz", s1[v]), Oψ1[v])

    Oψ2 = copy(ψ2)
    Oψ2[v] = apply(op("Sz", s2[v]), Oψ2[v])

    alg = "exact"
    expec_method1 =
        (inner(ψ1, Oψ1; alg) + inner(ψ2, Oψ2; alg) + 2 * inner(ψ1, Oψ2; alg)) /
        (norm_sqr(ψ1; alg) + norm_sqr(ψ2; alg) + 2 * inner(ψ1, ψ2; alg))
    expec_method2 = inner(ψ12, Oψ12; alg) / norm_sqr(ψ12; alg)

    @test expec_method1 ≈ expec_method2
end

#
# This test is a regression test for an
# issue where summing two product states
# results in incorrect fluxes of the
# output state's tensors
#
@testset "Sum Product States" begin
    g = named_comb_tree((2, 2))
    sites = siteinds("S=1/2", g; conserve_qns = true)

    verts = collect(vertices(g))

    state1 = Dict{Tuple, String}()
    for (j, v) in enumerate(verts)
        state1[v] = isodd(j) ? "Up" : "Dn"
    end
    ψ1 = ttn(state1, sites)

    state2 = Dict{Tuple, String}()
    for (j, v) in enumerate(vertices(g))
        state2[v] = isodd(j) ? "Dn" : "Up"
    end
    ψ2 = ttn(state2, sites)

    ϕ = ψ1 + ψ2

    for v in vertices(g)
        @test ITensors.allfluxequal(ITensors.tensor(ϕ[v]))
    end
end
end
