@eval module $(gensym())
using Compat: Compat
using Graphs: vertices, center
# Trigger package extension.
using ITensorNetworks:
  ITensorNetworks,
  BeliefPropagationCache,
  BoundaryMPSCache,
  QuadraticFormNetwork,
  ⊗,
  combine_linkinds,
  contract,
  contract_boundary_mps,
  contraction_sequence,
  eachtensor,
  environment,
  inner_network,
  linkinds_combiners,
  message,
  partitioned_tensornetwork,
  random_tensornetwork,
  scalar,
  siteinds,
  split_index,
  tensornetwork,
  update,
  update_factor,
  update_message,
  message_diff
using ITensors: ITensors, ITensor, combiner, dag, inds, inner, op, prime, random_itensor
using ITensorNetworks.ModelNetworks: ModelNetworks
using ITensors.NDTensors: array
using LinearAlgebra: eigvals, tr
using NamedGraphs: NamedEdge, NamedGraph, subgraph
using NamedGraphs.GraphsExtensions: rem_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedges
using SplitApplyCombine: group
using StableRNGs: StableRNG
using Test: @test, @testset
using OMEinsumContractionOrders
using LinearAlgebra: norm

@testset "boundarymps (eltype=$elt)" for elt in (
  Float32, Float64, Complex{Float32}, Complex{Float64}
)
  begin
    ITensors.disable_warn_order()
    mps_fit_kwargs = (; niters = 50, tolerance = 1e-10)

    #First a comb tree (which is still a planar graph)
    g = named_comb_tree((4,4))
    s = siteinds("S=1/2", g)
    χ = 2
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, elt, s; link_space=χ)
    ψIψ = QuadraticFormNetwork(ψ)
    vc = (first(center(g)), "operator")

    ρ_bp = contract(environment(ψIψ, [vc]; alg = "bp"); sequence = "automatic")
    ρ_bp /= tr(ρ_bp)

    ρ = subgraph(ψIψ, setdiff(vertices(ψIψ), [vc]))
    ρ_exact = contract(ρ; sequence = contraction_sequence(ρ; alg ="greedy"))
    ρ_exact /= tr(ρ_exact)

    ψIψ_boundaryMPS = BoundaryMPSCache(ψIψ; message_rank = χ*χ)
    ψIψ_boundaryMPS = update(ψIψ_boundaryMPS; mps_fit_kwargs)
    ρ_boundaryMPS = contract(environment(ψIψ_boundaryMPS, [vc]); sequence = "automatic")
    ρ_boundaryMPS /= tr(ρ_boundaryMPS)

    @test norm(ρ_boundaryMPS - ρ_exact) <= eps(real(elt))
    @test norm(ρ_boundaryMPS - ρ_bp) <= eps(real(elt))

    #Now a square graph with a few vertices missing for added complexity
    g = named_grid((5, 5))
    g = rem_vertices(g, [(2,2), (3,3)])
    s = siteinds("S=1/2", g)
    χ = 3
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, elt, s; link_space=χ)
    ψIψ = QuadraticFormNetwork(ψ)
    vc = (first(center(g)), "operator")
    ρ = subgraph(ψIψ, setdiff(vertices(ψIψ), [vc]))
    ρ_exact = contract(ρ; sequence = contraction_sequence(ρ; alg ="greedy"))
    ρ_exact /= tr(ρ_exact)

    ψIψ_boundaryMPS = BoundaryMPSCache(ψIψ; message_rank = χ*χ)
    ψIψ_boundaryMPS = update(ψIψ_boundaryMPS)
    ρ_boundaryMPS = contract(environment(ψIψ_boundaryMPS, [vc]); sequence = "automatic")
    ρ_boundaryMPS /= tr(ρ_boundaryMPS)

    @test norm(ρ_boundaryMPS - ρ_exact) <= 10*eps(real(elt))
    

  
  end
end
end
