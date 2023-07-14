using ITensorNetworks
using ITensorNetworks:
  approx_network_region,
  belief_propagation,
  get_environment,
  contract_inner,
  find_subgraph,
  message_tensors,
  neighbor_vertices,
  nested_graph_leaf_vertices,
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

function expect_bp(opname, v, ψ, mts)
  s = siteinds(ψ)
  ψψ = norm_network(ψ)
  numerator_network = approx_network_region(
    ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork(ITensor[apply(op(opname, s[v]), ψ[v])])
  )
  denominator_network = approx_network_region(ψψ, mts, [(v, 1)])
  return contract(numerator_network)[] / contract(denominator_network)[]
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
  mts = message_tensors(partition(ψψ, group(v -> v[1], vertices(ψψ))))
  mts = belief_propagation(
    ψψ, mts; contract_kwargs=(; alg="exact"), niters=50, target_precision=1e-5
  )
  for layer in eachindex(os)
    @show layer
    o⃗ = os[layer]
    for o in o⃗
      v⃗ = neighbor_vertices(ψ, o)
      for e in edges(mts)
        @assert order(only(mts[e])) == 2
        @assert order(only(mts[reverse(e)])) == 2
      end

      @assert length(v⃗) == 2
      v1, v2 = v⃗

      s1 = find_subgraph((v1, 1), mts)
      s2 = find_subgraph((v2, 1), mts)
      envs = get_environment(ψψ, mts, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])
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
      mts[s1] = ITensorNetwork(dictionary([(v1, 1) => ψψ[v1, 1], (v1, 2) => ψψ[v1, 2]]))
      mts[s2] = ITensorNetwork(dictionary([(v2, 1) => ψψ[v2, 1], (v2, 2) => ψψ[v2, 2]]))
      mts[s1 => s2] = ITensorNetwork(obs.singular_values)
      mts[s2 => s1] = ITensorNetwork(obs.singular_values)
    end
    if regauge
      println("regauge")
      mts = belief_propagation(
        ψψ, mts; contract_kwargs=(; alg="exact"), niters=50, target_precision=1e-5
      )
    end
  end
  return ψ, mts
end

function simple_update_vidal(os, ψ::ITensorNetwork; maxdim, regauge=false)
  println("Simple update, Vidal gauge")
  ψψ = norm_network(ψ)
  mts = message_tensors(partition(ψψ, group(v -> v[1], vertices(ψψ))))
  mts = belief_propagation(
    ψψ, mts; contract_kwargs=(; alg="exact"), niters=50, target_precision=1e-5
  )
  ψ, bond_tensors = vidal_gauge(ψ, mts)
  for layer in eachindex(os)
    @show layer
    o⃗ = os[layer]
    for o in o⃗
      v⃗ = neighbor_vertices(ψ, o)
      ψ, bond_tensors = apply(o, ψ, bond_tensors; maxdim, normalize=true)
    end
    if regauge
      println("regauge")
      ψ_symmetric, mts = vidal_to_symmetric_gauge(ψ, bond_tensors)
      ψψ = norm_network(ψ_symmetric)
      mts = belief_propagation(
        ψψ, mts; contract_kwargs=(; alg="exact"), niters=50, target_precision=1e-5
      )
      ψ, bond_tensors = vidal_gauge(ψ_symmetric, mts)
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
  ψ_bp, mts = simple_update_bp(
    os, ψ; maxdim=χ, variational_optimization_only, regauge, reduced
  )
  # ψ_bp, mts = vidal_to_symmetric_gauge(vidal_gauge(ψ_bp, mts)...)

  # Vidal SU
  ψ_vidal, bond_tensors = simple_update_vidal(os, ψ; maxdim=χ, regauge)
  ψ_vidal, mts_vidal = vidal_to_symmetric_gauge(ψ_vidal, bond_tensors)

  return ψ_bp, mts, ψ_vidal, mts_vidal
end
