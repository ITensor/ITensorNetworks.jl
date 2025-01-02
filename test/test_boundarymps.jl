@eval module $(gensym())
using Compat: Compat
using Graphs: vertices, center
# Trigger package extension.
using ITensorNetworks:
  ITensorNetworks,
  BeliefPropagationCache,
  BoundaryMPSCache,
  ITensorNetwork,
  QuadraticFormNetwork,
  VidalITensorNetwork,
  ⊗,
  combine_linkinds,
  contract,
  contract_boundary_mps,
  contraction_sequence,
  default_message_update,
  eachtensor,
  environment,
  inner_network,
  linkinds_combiners,
  message,
  messages,
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
using ITensors:
  ITensors, ITensor, combiner, dag, dim, inds, inner, normalize, op, prime, random_itensor
using ITensorNetworks.ModelNetworks: ModelNetworks
using ITensorNetworks.ITensorsExtensions: map_eigvals
using ITensors.NDTensors: array
using LinearAlgebra: eigvals, tr
using NamedGraphs: NamedEdge, NamedGraph, subgraph
using NamedGraphs.GraphsExtensions: rem_vertices
using NamedGraphs.NamedGraphGenerators:
  named_comb_tree, named_grid, named_hexagonal_lattice_graph
using NamedGraphs.PartitionedGraphs: PartitionEdge, PartitionVertex, partitionedges
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
    mps_fit_kwargs = (; niters=50, tolerance=1e-10)
    rng = StableRNG(1234)

    #First a comb tree (which is still a planar graph) and a flat tensor network
    g = named_comb_tree((4, 4))
    χ = 2
    tn = random_tensornetwork(rng, elt, g; link_space=χ)
    vc = first(center(g))

    ∂tn_∂vc_bp = contract(environment(tn, [vc]; alg="bp"); sequence="automatic")
    ∂tn_∂vc_bp /= norm(∂tn_∂vc_bp)

    ∂tn_∂vc = subgraph(tn, setdiff(vertices(tn), [vc]))
    ∂tn_∂vc_exact = contract(∂tn_∂vc; sequence=contraction_sequence(∂tn_∂vc; alg="greedy"))
    ∂tn_∂vc_exact /= norm(∂tn_∂vc_exact)

    #Orthogonal Boundary MPS
    tn_boundaryMPS = BoundaryMPSCache(tn; message_rank=1)
    tn_boundaryMPS = update(tn_boundaryMPS; mps_fit_kwargs)
    ∂tn_∂vc_boundaryMPS = contract(environment(tn_boundaryMPS, [vc]); sequence="automatic")
    ∂tn_∂vc_boundaryMPS /= norm(∂tn_∂vc_boundaryMPS)

    @test norm(∂tn_∂vc_boundaryMPS - ∂tn_∂vc_exact) <= 10 * eps(real(elt))
    @test norm(∂tn_∂vc_boundaryMPS - ∂tn_∂vc_bp) <= 10 * eps(real(elt))

    #Biorthogonal Boundary MPS
    tn_boundaryMPS = BoundaryMPSCache(tn; message_rank=1)
    tn_boundaryMPS = update(tn_boundaryMPS; alg="biorthogonal")
    ∂tn_∂vc_boundaryMPS = contract(environment(tn_boundaryMPS, [vc]); sequence="automatic")
    ∂tn_∂vc_boundaryMPS /= norm(∂tn_∂vc_boundaryMPS)

    @test norm(∂tn_∂vc_boundaryMPS - ∂tn_∂vc_exact) <= 10 * eps(real(elt))
    @test norm(∂tn_∂vc_boundaryMPS - ∂tn_∂vc_bp) <= 10 * eps(real(elt))

    #Now the norm tensor network of a square graph with a few vertices missing for added complexity
    g = named_grid((5, 5))
    g = rem_vertices(g, [(2, 2), (3, 3)])
    s = siteinds("S=1/2", g)
    χ = 3
    ψ = random_tensornetwork(rng, elt, s; link_space=χ)
    ψIψ = QuadraticFormNetwork(ψ)
    vc = (first(center(g)), "operator")
    ρ = subgraph(ψIψ, setdiff(vertices(ψIψ), [vc]))
    ρ_exact = contract(ρ; sequence=contraction_sequence(ρ; alg="greedy"))
    ρ_exact /= tr(ρ_exact)

    #Orthogonal Boundary MPS
    ψIψ_boundaryMPS = BoundaryMPSCache(ψIψ; message_rank=χ * χ)
    ψIψ_boundaryMPS = update(ψIψ_boundaryMPS)
    ρ_boundaryMPS = contract(environment(ψIψ_boundaryMPS, [vc]); sequence="automatic")
    ρ_boundaryMPS /= tr(ρ_boundaryMPS)

    @test norm(ρ_boundaryMPS - ρ_exact) <= 10 * eps(real(elt))

    #BiOrthogonal Boundary MPS
    ψIψ_boundaryMPS = BoundaryMPSCache(ψIψ; message_rank=χ * χ)
    ψIψ_boundaryMPS = update(ψIψ_boundaryMPS; alg="biorthogonal", maxiter=50)
    ρ_boundaryMPS = contract(environment(ψIψ_boundaryMPS, [vc]); sequence="automatic")
    ρ_boundaryMPS /= tr(ρ_boundaryMPS)

    @test norm(ρ_boundaryMPS - ρ_exact) <= 2e3 * eps(real(elt))

    #Now we test BP and orthogonal and biorthogonl Boundary MPS are equivalent when run from in the symmetric gauge
    g = named_hexagonal_lattice_graph(3, 3)
    s = siteinds("S=1/2", g)
    χ = 2
    ψ = random_tensornetwork(rng, elt, s; link_space=χ)

    #Move wavefunction to symmetric gauge, enforce posdefness for complex number stability
    function make_posdef(A::ITensor)
      return map_eigvals(
        x -> abs(real(x)), A, first(inds(A)), last(inds(A)); ishermitian=true
      )
    end
    message_update_f = ms -> make_posdef.(default_message_update(ms))
    ψ_vidal = VidalITensorNetwork(
      ψ; cache_update_kwargs=(; message_update=message_update_f, maxiter=100, tol=1e-16)
    )
    cache_ref = Ref{BeliefPropagationCache}()
    ψ_symm = ITensorNetwork(ψ_vidal; (cache!)=cache_ref)
    bp_cache = cache_ref[]

    #Do Orthogonal Boundary MPS
    ψIψ_boundaryMPS = BoundaryMPSCache(QuadraticFormNetwork(ψ_symm); message_rank=1)
    ψIψ_boundaryMPS = update(ψIψ_boundaryMPS; mps_fit_kwargs)

    for pe in keys(messages(bp_cache))
      m_boundarymps = only(message(ψIψ_boundaryMPS, pe))
      #Prune the dimension 1 virtual index from boundary MPS message tensor
      m_boundarymps =
        m_boundarymps * ITensor(1.0, filter(i -> dim(i) == 1, inds(m_boundarymps)))
      m_bp = only(message(bp_cache, pe))
      m_bp /= tr(m_bp)
      m_boundarymps /= tr(m_boundarymps)
      @test norm(m_bp - m_boundarymps) <= 1e-4
    end

    #Do Biorthogonal Boundary MPS
    ψIψ_boundaryMPS = BoundaryMPSCache(QuadraticFormNetwork(ψ_symm); message_rank=1)
    ψIψ_boundaryMPS = update(ψIψ_boundaryMPS; alg="biorthogonal")

    for pe in keys(messages(bp_cache))
      m_boundarymps = only(message(ψIψ_boundaryMPS, pe))
      #Prune the dimension 1 virtual index from boundary MPS message tensor
      m_boundarymps =
        m_boundarymps * ITensor(1.0, filter(i -> dim(i) == 1, inds(m_boundarymps)))
      m_bp = only(message(bp_cache, pe))
      m_bp /= tr(m_bp)
      m_boundarymps /= tr(m_boundarymps)
      @test norm(m_bp - m_boundarymps) <= 1e-4
    end
  end
end
end
