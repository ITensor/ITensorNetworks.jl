@eval module $(gensym())
using Adapt: Adapt, adapt
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensorNetworks: random_tensornetwork, siteinds
using ITensors: ITensors
using Test: @test, @testset

struct SinglePrecisionAdaptor end
single_precision(::Type{<:AbstractFloat}) = Float32
single_precision(type::Type{<:Complex}) = complex(single_precision(real(type)))
Adapt.adapt_storage(::SinglePrecisionAdaptor, x) = single_precision(eltype(x)).(x)

@testset "Test ITensorNetworksAdaptExt (eltype=$elt)" for elt in (
  Float32, Float64, Complex{Float32}, Complex{Float64}
)
  g = named_grid((2, 2))
  s = siteinds("S=1/2", g)
  tn = random_tensornetwork(elt, s)
  @test ITensors.scalartype(tn) === elt
  tn′ = adapt(SinglePrecisionAdaptor(), tn)
  @test ITensors.scalartype(tn′) === single_precision(elt)
end
end
