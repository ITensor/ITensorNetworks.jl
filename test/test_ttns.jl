using DataGraphs: vertex_data
using Graphs: vertices
using ITensorNetworks:
    ITensorNetwork, TreeTensorNetwork, contract, ortho_region, orthogonalize, siteinds, ttn
using ITensors: @disable_warn_order, random_itensor
using LinearAlgebra: norm
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Random: shuffle
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "TTN Basics" begin
    # random comb tree
    rng = StableRNG(1234)
    tooth_lengths = rand(rng, 1:3, rand(rng, 2:4))
    c = named_comb_tree(tooth_lengths)
    # specify random site dimension on every site
    dmap = v -> rand(rng, 1:3)
    is = siteinds(dmap, c)
    # specify random linear vertex ordering of graph vertices
    vertex_order = shuffle(rng, collect(vertices(c)))

    @testset "Construct TTN from ITensor or Array" begin
        cutoff = 1.0e-10
        # create random ITensor with these indices
        rng = StableRNG(1234)
        S = random_itensor(rng, vertex_data(is)...)
        # dense TTN constructor from IndsNetwork
        @disable_warn_order s1 = ttn(S, is; cutoff)
        root_vertex = only(ortho_region(s1))
        @disable_warn_order begin
            S1 = contract(s1, root_vertex)
        end
        @test norm(S - S1) < 1.0e2 * cutoff
    end

    @testset "Convert ITN <-> TTN" begin
        g = named_comb_tree((3, 2))
        sites = siteinds("S=1/2", g)

        psi = ttn(sites)  # zero-initialised
        psi = ttn(v -> "Up", sites)  # product state

        itn = ITensorNetwork(psi)  # TTN → ITensorNetwork
        @test vertex_data(itn) == vertex_data(psi.tensornetwork)
        @test !(itn === psi.tensornetwork)
        @test vertex_data(TreeTensorNetwork(itn)) == vertex_data(psi)
    end

    @testset "Ortho" begin
        g = named_comb_tree((3, 2))
        sites = siteinds("S=1/2", g)

        psi = ttn(sites)  # zero-initialised
        psi = ttn(v -> "Up", sites)  # product state

        v1 = collect(vertices(psi))[1]
        v2 = collect(vertices(psi))[2]

        @test collect(ortho_region(orthogonalize(psi, v1))) == [v1]
        @test collect(ortho_region(orthogonalize(psi, [v1, v2]))) == [v1, v2]
    end
end
