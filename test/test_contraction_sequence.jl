@eval module $(gensym())
using EinExprs: Exhaustive, Greedy, HyPar
using ITensorNetworks:
  contraction_sequence, norm_sqr_network, random_tensornetwork, siteinds
using ITensors: ITensors, contract
using NamedGraphs.NamedGraphGenerators: named_grid
using OMEinsumContractionOrders: OMEinsumContractionOrders
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "contraction_sequence" begin
  ITensors.@disable_warn_order begin
    dims = (2, 3)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    χ = 10
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, s; link_space=χ)
    tn = norm_sqr_network(ψ)
    seq_optimal = contraction_sequence(tn; alg="optimal")
    res_optimal = contract(tn; sequence=seq_optimal)[]
    seq_greedy = contraction_sequence(tn; alg="greedy")
    res_greedy = contract(tn; sequence=seq_greedy)[]
    seq_tree_sa = contraction_sequence(tn; alg="tree_sa")
    res_tree_sa = contract(tn; sequence=seq_tree_sa)[]
    seq_sa_bipartite = contraction_sequence(tn; alg="sa_bipartite")
    res_sa_bipartite = contract(tn; sequence=seq_sa_bipartite)[]
    seq_einexprs_exhaustive = contraction_sequence(
      tn; alg="einexpr", optimizer=Exhaustive()
    )
    res_einexprs_exhaustive = contract(tn; sequence=seq_einexprs_exhaustive)[]
    seq_einexprs_greedy = contraction_sequence(tn; alg="einexpr", optimizer=Greedy())
    res_einexprs_greedy = contract(tn; sequence=seq_einexprs_exhaustive)[]
    @test res_greedy ≈ res_optimal
    @test res_tree_sa ≈ res_optimal
    @test res_sa_bipartite ≈ res_optimal
    @test res_einexprs_exhaustive ≈ res_optimal
    @test res_einexprs_greedy ≈ res_optimal

    if !Sys.iswindows()
      # KaHyPar doesn't work on Windows
      # https://github.com/kahypar/KaHyPar.jl/issues/9
      using Pkg
      Pkg.add("KaHyPar"; io=devnull)
      using KaHyPar
      seq_kahypar_bipartite = contraction_sequence(
        tn; alg="kahypar_bipartite", sc_target=200
      )
      Pkg.rm("KaHyPar"; io=devnull)
      res_kahypar_bipartite = contract(tn; sequence=seq_kahypar_bipartite)[]
      @test res_optimal ≈ res_kahypar_bipartite
      seq_einexprs_kahypar = contraction_sequence(tn; alg="einexpr", optimizer=HyPar())
      res_einexprs_kahypar = contract(tn; sequence=seq_einexprs_kahypar)[]
      @test res_einexprs_kahypar ≈ res_optimal
    end
  end
end
end
