using ITensorNetworks
using ITensorNetworks:
  approx_network_region,
  belief_propagation,
  get_environment,
  contract_inner,
  message_tensors,
  neighbor_vertices,
  symmetric_gauge,
  vidal_gauge,
  vidal_to_symmetric_gauge,
  norm_network
using Test
using Compat
using Dictionaries
using ITensors
using Metis
using NamedGraphs
using Observers
using Random
using LinearAlgebra
using SplitApplyCombine
using OMEinsumContractionOrders

function expect_bp(opname, v, ψ, pψψ, mts)
  s = siteinds(ψ)
  numerator_tensors = approx_network_region(
    pψψ, mts, [(v, 1)]; verts_tensors=ITensor[apply(op(opname, s[v]), ψ[v])]
  )
  denominator_tensors = approx_network_region(pψψ, mts, [(v, 1)])
  return contract(numerator_tensors)[] / contract(denominator_tensors)[]
end

function vertex_array(ψ, v, v⃗ⱼ)
  indsᵥ = unioninds((linkinds(ψ, v => vⱼ) for vⱼ in v⃗ⱼ)...)
  indsᵥ = unioninds(siteinds(ψ, v), indsᵥ)
  ψᵥ = ψ[v]
  ψᵥ /= norm(ψᵥ)
  return array(permute(ψᵥ, indsᵥ))
end

function simple_update_bp(
  os,
  ψ::ITensorNetwork;
  maxdim,
  variational_optimization_only=false,
  regauge=false,
  reduced=true,
)
  println("Simple update, BP")
  ψψ = norm_network(ψ)
  pψψ = PartitionedGraph(ψψ, group(v -> v[1], vertices(ψψ)))
  mts = belief_propagation(
    pψψ; contract_kwargs=(; alg="exact"), niters=50, target_precision=1e-5
  )
  edges = PartitionEdge.(NamedGraphs.edges(partitioned_graph(pψψ)))
  for layer in eachindex(os)
    @show layer
    o⃗ = os[layer]
    for o in o⃗
      v⃗ = neighbor_vertices(ψ, o)
      for e in edges
        @assert order(only(mts[e])) == 2
        @assert order(only(mts[PartitionEdge(reverse(NamedGraphs.parent(e)))])) == 2
      end

      @assert length(v⃗) == 2
      v1, v2 = v⃗

      pe = NamedGraphs.partition_edge(pψψ, NamedEdge((v1, 1) => (v2, 1)))
      envs = get_environment(pψψ, mts, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])
      obs = observer()
      # TODO: Make a version of `apply` that accepts message tensors,
      # and computes the environment and does the message tensor update of the bond internally.
      ψ = apply(
        o,
        ψ;
        envs,
        (observer!)=obs,
        maxdim,
        normalize=true,
        variational_optimization_only,
        nfullupdatesweeps=20,
        symmetrize=true,
        reduced,
      )
      S = only(obs.singular_values)
      S /= norm(S)

      # Update message tensor
      ψψ = norm_network(ψ)
      pψψ = PartitionedGraph(ψψ, group(v -> v[1], vertices(ψψ)))
      mts[pe] = dense.(obs.singular_values)
      mts[PartitionEdge(reverse(NamedGraphs.parent(pe)))] = dense.(obs.singular_values)
    end
    if regauge
      println("regauge")
      mts = belief_propagation(
        pψψ, mts; contract_kwargs=(; alg="exact"), niters=50, target_precision=1e-5
      )
    end
  end
  return ψ, pψψ, mts
end

function simple_update_vidal(os, ψ::ITensorNetwork; maxdim, regauge=false)
  println("Simple update, Vidal gauge")
  ψψ = norm_network(ψ)
  pψψ = PartitionedGraph(ψψ, group(v -> v[1], vertices(ψψ)))
  mts = belief_propagation(
    pψψ; contract_kwargs=(; alg="exact"), niters=50, target_precision=1e-5
  )
  ψ, bond_tensors = vidal_gauge(ψ, pψψ, mts)
  for layer in eachindex(os)
    @show layer
    o⃗ = os[layer]
    for o in o⃗
      v⃗ = neighbor_vertices(ψ, o)
      ψ, bond_tensors = apply(o, ψ, bond_tensors; maxdim, normalize=true)
    end
    if regauge
      println("regauge")
      ψ_symmetric, pψψ_symmetric, mts = vidal_to_symmetric_gauge(ψ, bond_tensors)
      mts = belief_propagation(
        pψψ_symmetric,
        mts;
        contract_kwargs=(; alg="exact"),
        niters=50,
        target_precision=1e-5,
      )
      ψ, bond_tensors = vidal_gauge(ψ_symmetric, pψψ_symmetric, mts)
    end
  end
  return ψ, bond_tensors
end

function main(;
  seed=5623,
  graph,
  opname,
  dims,
  χ,
  nlayers,
  variational_optimization_only=false,
  regauge=false,
  reduced=true,
)
  Random.seed!(seed)
  n = prod(dims)
  g = graph(dims)
  s = siteinds("S=1/2", g)
  ψ = randomITensorNetwork(s; link_space=χ)
  es = edges(g)
  os = [
    [op(opname, s[src(e)]..., s[dst(e)]...; eltype=Float64) for e in es] for _ in 1:nlayers
  ]

  # BP SU
  ψ_bp, pψψ_bp, mts_bp = simple_update_bp(
    os, ψ; maxdim=χ, variational_optimization_only, regauge, reduced
  )
  # ψ_bp, mts = vidal_to_symmetric_gauge(vidal_gauge(ψ_bp, mts)...)

  # Vidal SU
  ψ_vidal, bond_tensors = simple_update_vidal(os, ψ; maxdim=χ, regauge)
  ψ_vidal, pψψ_vidal, mts_vidal = vidal_to_symmetric_gauge(ψ_vidal, bond_tensors)

  return ψ_bp, pψψ_bp, mts_bp, ψ_vidal, pψψ_vidal, mts_vidal
end
