using ITensorNetworks: AbstractITensorNetwork, ITensorNetwork
using ITensorVisualizationBase: ITensorVisualizationBase
using ITensors: ITensor, Index
using Test: @test, @testset

@testset "ITensorNetworksITensorVisualizationBaseExt" begin
    i, j, k = Index(2, "i"), Index(2, "j"), Index(2, "k")
    tn = ITensorNetwork([ITensor(i, j), ITensor(j, k)])
    @test hasmethod(ITensorVisualizationBase.visualize, Tuple{AbstractITensorNetwork})
    @test isnothing(ITensorVisualizationBase.visualize(tn))
    @test isnothing(ITensorVisualizationBase.visualize(tn; vertex_labels_prefix = "v"))
end
