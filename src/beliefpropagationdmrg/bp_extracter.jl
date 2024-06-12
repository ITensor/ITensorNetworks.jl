using ITensors: scalartype
using ITensorNetworks:
  ket_vertices, bra_vertices, tensornetwork, default_message_update, operator_network
using ITensorNetworks.ITensorsExtensions: map_eigvals

function update_caches_effective_environments(
  state::ITensorNetwork,
  ψOψ_bpcs::Vector{<:BeliefPropagationCache},
  ψIψ_bpc::BeliefPropagationCache,
  region,
)
  operators = operator_network.(tensornetwork.(ψOψ_bpcs))

  ψIψ_bpc_mts = messages(ψIψ_bpc)
  new_ψOψ_bpcs = BeliefPropagationCache[]

  for (ψOψ_bpc, operator) in zip(ψOψ_bpcs, operators)
    new_ψOψ_bpc = copy(ψOψ_bpc)
    ptn = partitioned_tensornetwork(new_ψOψ_bpc)
    edge_seq = reduce(
      vcat, [post_order_dfs_edges(underlying_graph(operator), v) for v in region]
    )
    broken_edges = setdiff(edges(state), edges(operator))
    partition_broken_edges = PartitionEdge.(broken_edges)
    partition_broken_edges = vcat(partition_broken_edges, reverse.(partition_broken_edges))
    mts = messages(new_ψOψ_bpc)
    for pe in partition_broken_edges
      set!(mts, pe, copy(ψIψ_bpc_mts[pe]))
    end
    partition_edge_seq = unique(PartitionEdge.(edge_seq))
    new_ψOψ_bpc = update(
      new_ψOψ_bpc,
      partition_edge_seq;
      message_update=tns -> default_message_update(tns; normalize=false),
    )
    push!(new_ψOψ_bpcs, new_ψOψ_bpc)
  end

  return state, new_ψOψ_bpcs, ψIψ_bpc
end

function bp_extracter(
  ψ::AbstractITensorNetwork,
  ψOψ_bpcs::Vector{<:BeliefPropagationCache},
  ψIψ_bpc::BeliefPropagationCache,
  region;
  regularization=10 * eps(scalartype(ψ)),
  ishermitian=true,
)
  ψ, ψOψ_bpcs, ψIψ_bpc = update_caches_effective_environments(ψ, ψOψ_bpcs, ψIψ_bpc, region)

  form_network = tensornetwork(ψIψ_bpc)
  form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region),
  bra_vertices(form_network, region)

  ∂ψOψ_bpc_∂rs = environment.(ψOψ_bpcs, ([form_ket_vertices; form_bra_vertices],))
  state = prod([ψ[v] for v in region])
  messages = environment(ψIψ_bpc, partitionvertices(ψIψ_bpc, form_ket_vertices))
  f_sqrt = sqrt ∘ (x -> x + regularization)
  f_inv_sqrt = inv ∘ sqrt ∘ (x -> x + regularization)
  sqrt_mts =
    map_eigvals.(
      (f_sqrt,), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian
    )
  inv_sqrt_mts =
    map_eigvals.(
      (f_inv_sqrt,), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian
    )

  return state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts
end
