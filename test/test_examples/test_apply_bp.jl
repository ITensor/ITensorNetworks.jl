using ITensorNetworks
using Suppressor
using Test

include(joinpath(pkgdir(ITensorNetworks), "examples", "apply", "apply_bp", "apply_bp.jl"))

@testset "Test apply_bp example" begin
  opnames = ["Id", "RandomUnitary"]
  graphs = (named_comb_tree, named_grid)
  dims = (6, 6)
  @testset "$opname, $graph" for opname in opnames, graph in graphs
    ψ_bp, pψψ_bp, mts_bp, ψ_vidal, pψψ_vidal, mts_vidal = @suppress main(;
      seed=1234,
      opname,
      graph,
      dims,
      χ=2,
      nlayers=2,
      variational_optimization_only=false,
      regauge=false,
      reduced=true,
    )
    v = dims .÷ 2
    sz_bp = expect_bp("Sz", v, ψ_bp, pψψ_bp, mts_bp)
    sz_vidal = expect_bp("Sz", v, ψ_vidal, pψψ_vidal, mts_vidal)
    @test sz_bp ≈ sz_vidal rtol = 1e-5
  end
end
