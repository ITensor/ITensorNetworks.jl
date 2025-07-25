using Graphs: vertices
using NamedGraphs: AbstractNamedGraph, NamedEdge
using NamedGraphs.PartitionedGraphs: partitionedges
using Printf: @printf
using ConstructionBase: setproperties

@kwdef mutable struct FittingProblem{State<:AbstractBeliefPropagationCache}
  state::State
  ket_graph::AbstractNamedGraph
  overlap::Number = 0
  gauge_region
end

state(F::FittingProblem) = F.state
ket_graph(F::FittingProblem) = F.ket_graph
overlap(F::FittingProblem) = F.overlap
gauge_region(F::FittingProblem) = F.gauge_region

function set_state(F::FittingProblem, state)
  FittingProblem(state, F.ket_graph, F.overlap, F.gauge_region)
end
function set_overlap(F::FittingProblem, overlap)
  FittingProblem(F.state, F.ket_graph, overlap, F.gauge_region)
end

function ket(F::FittingProblem)
  ket_vertices = vertices(ket_graph(F))
  return first(induced_subgraph(tensornetwork(state(F)), ket_vertices))
end

function extract(problem::FittingProblem, region_iterator; sweep, kws...)
  region = current_region(region_iterator)
  prev_region = gauge_region(problem)
  tn = state(problem)
  path = edge_sequence_between_regions(ket_graph(problem), prev_region, region)
  tn = gauge_walk(Algorithm("orthogonalize"), tn, path)
  pe_path = partitionedges(partitioned_tensornetwork(tn), path)
  tn = update(
    Algorithm("bp"), tn, pe_path; message_update_function_kwargs=(; normalize=false)
  )
  local_tensor = environment(tn, region)
  sequence = contraction_sequence(local_tensor; alg="optimal")
  local_tensor = dag(contract(local_tensor; sequence))
  #problem, local_tensor = subspace_expand(problem, local_tensor, region; sweep, kws...)
  return setproperties(problem; state=tn, gauge_region=region), local_tensor
end

function update(F::FittingProblem, local_tensor, region; outputlevel, kws...)
  n = (local_tensor * dag(local_tensor))[]
  F = set_overlap(F, n / sqrt(n))
  if outputlevel >= 2
    @printf("  Region %s: squared overlap = %.12f\n", region, overlap(F))
  end
  return F, local_tensor
end

function region_plan(F::FittingProblem; nsites, sweep_kwargs...)
  return euler_sweep(ket_graph(F); nsites, sweep_kwargs...)
end

function fit_tensornetwork(
  overlap_network,
  args...;
  nsweeps=25,
  nsites=1,
  outputlevel=0,
  extract_kwargs=(;),
  update_kwargs=(;),
  insert_kwargs=(;),
  normalize=true,
  kws...,
)
  bpc = BeliefPropagationCache(overlap_network, args...)
  ket_graph = first(
    induced_subgraph(underlying_graph(overlap_network), ket_vertices(overlap_network))
  )
  init_prob = FittingProblem(;
    ket_graph, state=bpc, gauge_region=collect(vertices(ket_graph))
  )

  insert_kwargs = (; insert_kwargs..., normalize, set_orthogonal_region=false)
  common_sweep_kwargs = (; nsites, outputlevel, update_kwargs, insert_kwargs)
  kwargs_array = [(; common_sweep_kwargs..., sweep=s) for s in 1:nsweeps]
  sweep_iter = sweep_iterator(init_prob, kwargs_array)
  converged_prob = sweep_solve(sweep_iter; outputlevel, kws...)
  return rename_vertices(inv_vertex_map(overlap_network), ket(converged_prob))
end

function fit_tensornetwork(tn, init_state, args...; kwargs...)
  return fit_tensornetwork(inner_network(tn, init_state), args; kwargs...)
end

#function truncate(tn; maxdim=default_maxdim(), cutoff=default_cutoff(), kwargs...)
#  init_state = ITensorNetwork(
#    v -> inds -> delta(inds), siteinds(tn); link_space=maxdim
#  )
#  overlap_network = inner_network(tn, init_state)
#  insert_kwargs = (; trunc=(; cutoff, maxdim))
#  return fit_tensornetwork(overlap_network; insert_kwargs, kwargs...)
#end

function ITensors.apply(
  A::ITensorNetwork,
  x::ITensorNetwork;
  maxdim=default_maxdim(),
  cutoff=default_cutoff(),
  kwargs...,
)
  init_state = ITensorNetwork(v -> inds -> delta(inds), siteinds(x); link_space=maxdim)
  overlap_network = inner_network(x, A, init_state)
  insert_kwargs = (; trunc=(; cutoff, maxdim))
  return fit_tensornetwork(overlap_network; insert_kwargs, kwargs...)
end
