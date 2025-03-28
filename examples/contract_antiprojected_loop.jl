using NamedGraphs.GraphsExtensions:
  add_edge, add_vertex, src, dst, vertices, subgraph, induced_subgraph, reverse, edges
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs: NamedEdge
using NamedGraphs.PartitionedGraphs:
  partitionvertex, PartitionEdge, partitionedges, partitionvertices
using ITensors: siteinds
using ITensorNetworks:
  random_tensornetwork,
  BeliefPropagationCache,
  QuadraticFormNetwork,
  update,
  environment,
  partitioned_tensornetwork,
  tensornetwork,
  eachtensor,
  ITensorNetwork,
  contraction_sequence,
  linkinds,
  message,
  factor,
  update_factor,
  messages,
  region_scalar,
  vertex_scalars,
  edge_scalars,
  norm_sqr_network,
  update_factors
using ITensors:
  ITensor,
  contract,
  sim,
  replaceinds,
  combiner,
  combinedind,
  delta,
  Index,
  inds,
  replaceind,
  noprime,
  norm
using OMEinsumContractionOrders: OMEinsumContractionOrders
using Dictionaries: Dictionary, set!
using LinearAlgebra: norm, LinearAlgebra, normalize, dot
using SplitApplyCombine: group

using Random

function normalize_messages(bp_cache::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
  bp_cache = copy(bp_cache)
  mts = messages(bp_cache)
  for pe in pes
    me, mer = only(mts[pe]), only(mts[reverse(pe)])
    set!(mts, pe, ITensor[me / norm(me)])
    set!(mts, reverse(pe), ITensor[mer / norm(mer)])
    n = region_scalar(bp_cache, pe)
    set!(mts, pe, ITensor[(1 / sqrt(n)) * me])
    set!(mts, reverse(pe), ITensor[(1 / sqrt(n)) * mer])
  end
  return bp_cache
end

function normalize_message(bp_cache::BeliefPropagationCache, pe::PartitionEdge)
  return normalize_messages(bp_cache, PartitionEdge[pe])
end

function normalize_messages(bp_cache::BeliefPropagationCache)
  return normalize_messages(bp_cache, partitionedges(partitioned_tensornetwork(bp_cache)))
end

function true_delta(row_inds::Vector{<:Index}, col_inds::Vector{<:Index})
  row_c, col_c = combiner(row_inds), combiner(col_inds)
  td = delta(combinedind(row_c), combinedind(col_c))
  return td * col_c * row_c
end

function anti_project_edges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
  bpc = copy(bpc)
  antiprojectors = ITensor[]
  for pe in pes
    indices = linkinds(bpc, pe)
    me, mer = only(message(bpc, pe)), only(message(bpc, reverse(pe)))
    dual_indices = [sim(noprime(ind)) for ind in indices]
    dual_inds_dict = Dictionary(indices, dual_indices)
    me = replaceinds(me, indices, dual_indices)
    anti_proj = true_delta(indices, dual_indices) - me * mer
    push!(antiprojectors, anti_proj)
    @assert inds(anti_proj) == vcat(indices, dual_indices)
    for v in vertices(bpc, dst(pe))
      ψv = only(factors(bpc, [v]))
      c_inds = intersect(inds(ψv), indices)
      for c in c_inds
        ψv = replaceind(ψv, c, dual_inds_dict[c])
      end
      bpc = update_factor(bpc, v, ψv)
    end
  end
  return bpc, antiprojectors
end

function LinearAlgebra.normalize(
  ψ::ITensorNetwork; cache_update_kwargs=(; maxiter=30, tol=1e-12)
)
  ψψ = norm_sqr_network(ψ)
  ψψ_bpc = BeliefPropagationCache(ψψ, group(v -> first(v), vertices(ψψ)))
  ψ, ψψ_bpc = normalize(ψ, ψψ_bpc; cache_update_kwargs)
  return ψ, ψψ_bpc
end

function LinearAlgebra.normalize(
  ψ::ITensorNetwork,
  ψAψ_bpc::BeliefPropagationCache;
  cache_update_kwargs=default_cache_update_kwargs(ψAψ_bpc),
  update_cache=true,
  sf::Float64=1.0,
)
  ψ = copy(ψ)
  if update_cache
    ψAψ_bpc = update(ψAψ_bpc; cache_update_kwargs...)
  end
  ψAψ_bpc = normalize_messages(ψAψ_bpc)
  ψψ = tensornetwork(ψAψ_bpc)

  for v in vertices(ψ)
    v_ket, v_bra = (v, "ket"), (v, "bra")
    pv = only(partitionvertices(ψAψ_bpc, [v_ket]))
    vn = region_scalar(ψAψ_bpc, pv)
    state = copy(ψψ[v_ket]) / sqrt(sf * vn)
    state_dag = copy(ψψ[v_bra]) / sqrt(sf * vn)
    vertices_states = Dictionary([v_ket, v_bra], [state, state_dag])
    ψAψ_bpc = update_factors(ψAψ_bpc, vertices_states)
    ψ[v] = state
  end

  return ψ, ψAψ_bpc
end

Random.seed!(1234)

g = named_grid((2, 2))
g = add_vertex(g, (2, 3))
g = add_edge(g, NamedEdge((2, 3) => (2, 2)))

s = siteinds("S=1/2", g)
ψ = random_tensornetwork(s; link_space=2)
ψ, _ = normalize(ψ)
ψIψ_bpc = BeliefPropagationCache(QuadraticFormNetwork(ψ))
ψIψ_bpc = update(ψIψ_bpc; maxiter=20)
ψIψ_bpc = normalize_messages(ψIψ_bpc)
bp_norm = prod(vertex_scalars(ψIψ_bpc))
pg = partitioned_tensornetwork(ψIψ_bpc)

loop =
  PartitionEdge.([
    NamedEdge((1, 1) => (1, 2)),
    NamedEdge((1, 2) => (2, 2)),
    NamedEdge((2, 2) => (2, 1)),
    NamedEdge((2, 1) => (1, 1)),
  ])
partition_vertices_in_loop = unique(vcat(src.(loop), dst.(loop)))

incoming_messages = environment(ψIψ_bpc, partition_vertices_in_loop)
bpc, antiprojectors = anti_project_edges(ψIψ_bpc, loop)
tn = factors(bpc, vertices(bpc, partition_vertices_in_loop))

all_tensors = vcat(vcat(tn, antiprojectors), incoming_messages)
seq = contraction_sequence(all_tensors; alg="sa_bipartite")
loop_correction = contract(all_tensors; sequence=seq)[]

true_contraction = bp_norm + loop_correction
@show true_contraction

ψIψ = QuadraticFormNetwork(ψ)
all_tensors = [ψIψ[v] for v in vertices(ψIψ)]
seq = contraction_sequence(all_tensors; alg="sa_bipartite")
actual_contraction = contract(all_tensors; sequence=seq)[]
@show actual_contraction
