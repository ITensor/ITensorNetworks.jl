using DataGraphs: vertex_data
using Graphs: vertices
using ITensorNetworks:
    ITensorNetwork, TreeTensorNetwork, ortho_region, orthogonalize, siteinds
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using StableRNGs: StableRNG
using Test: @test, @testset
include("utils.jl")
@testset "TTN Basics" begin
    @testset "Convert ITN <-> TTN" begin
        g = named_comb_tree((3, 2))
        sites = siteinds("S=1/2", g)

        rng = StableRNG(1234)
        psi = TreeTensorNetwork(random_tensornetwork(rng, sites))  # random
        psi = TreeTensorNetwork(productstate(v -> "Up", sites))  # product state

        itn = ITensorNetwork(psi)  # TTN → ITensorNetwork
        @test vertex_data(itn) == vertex_data(psi)
        @test !(vertex_data(itn) === vertex_data(psi))
        @test vertex_data(TreeTensorNetwork(itn)) == vertex_data(psi)
    end

    @testset "Ortho" begin
        g = named_comb_tree((3, 2))
        sites = siteinds("S=1/2", g)

        rng = StableRNG(1234)
        psi = TreeTensorNetwork(random_tensornetwork(rng, sites))  # random
        psi = TreeTensorNetwork(productstate(v -> "Up", sites))  # product state

        v1 = collect(vertices(psi))[1]
        v2 = collect(vertices(psi))[2]

        @test collect(ortho_region(orthogonalize(psi, v1))) == [v1]
        @test collect(ortho_region(orthogonalize(psi, [v1, v2]))) == [v1, v2]
    end
end
