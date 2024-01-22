using ITensorNetworks
using Suppressor
using Test

include(joinpath(pkgdir(ITensorNetworks), "examples", "belief_propagation", "sqrt_bp.jl"))

@testset "Test sqrt_bp example" begin
  (; sz_bp, sz_sqrt_bp) = main(; n=8, niters=10, β=0.28, h=0.5)
  @test sz_bp ≈ sz_sqrt_bp
end
