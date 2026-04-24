@eval module $(gensym())
using Graphs: vertices
using ITensorNetworks: ITensorNetworks, OpSum, siteinds, ttn
using ITensors: ITensors
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @testset

@testset "OpSum to TTN converter" begin
    @testset "Multiple onsite terms (regression test for issue #62)" begin
        auto_fermion_enabled = ITensors.using_auto_fermion()
        if !auto_fermion_enabled
            ITensors.enable_auto_fermion()
        end
        grid_dims = (2, 1)
        g = named_grid(grid_dims)
        s = siteinds("S=1/2", g)

        os1 = OpSum()
        os1 += 1.0, "Sx", (1, 1)
        os2 = OpSum()
        os2 += 1.0, "Sy", (1, 1)
        H1 = ttn(os1, s)
        H2 = ttn(os2, s)
        H3 = ttn(os1 + os2, s)

        @test H1 + H2 ≈ H3 rtol = 1.0e-6
        if auto_fermion_enabled
            ITensors.enable_auto_fermion()
        end
    end
end
end
