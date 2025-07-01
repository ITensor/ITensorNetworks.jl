import ITensorNetworks as itn
import ITensors as it
import Graphs: vertices
import NamedGraphs.PartitionedGraphs as npg
using NamedGraphs: AbstractNamedGraph, NamedEdge
using Printf

@kwdef mutable struct FittingProblem{State<:itn.AbstractBeliefPropagationCache}
  state::State
  ket_graph::AbstractNamedGraph
  overlap::Number = 0
  gauge_region
end

overlap(F::FittingProblem) = F.overlap
state(F::FittingProblem) = F.state
ket_graph(F::FittingProblem) = F.ket_graph
gauge_region(F::FittingProblem) = F.gauge_region

function ket(F::FittingProblem)
  ket_vertices = vertices(ket_graph(F))
  return first(itn.induced_subgraph(itn.tensornetwork(state(F)), ket_vertices))
end

function extracter(problem::FittingProblem, region_iterator; sweep, kws...)
  region = current_region(region_iterator)
  prev_region = gauge_region(problem)
  tn = state(problem)
  path = itn.edge_sequence_between_regions(ket_graph(problem), prev_region, region)
  tn = itn.gauge_walk(itn.Algorithm("orthogonalize"), tn, path)
  pe_path = npg.partitionedges(itn.partitioned_tensornetwork(tn), path)
  tn = itn.update(
    itn.Algorithm("bp"), tn, pe_path; message_update_function_kwargs=(; normalize=false)
  )
  local_tensor = itn.environment(tn, region)
  sequence = itn.contraction_sequence(local_tensor; alg="optimal")
  local_tensor = dag(it.contract(local_tensor; sequence))
  #problem, local_tensor = subspace_expand(problem, local_tensor, region; sweep, kws...)
  return setproperties(problem; state=tn, gauge_region=region), local_tensor
end

function updater(F::FittingProblem, local_tensor, region; outputlevel, kws...)
  n = (local_tensor * dag(local_tensor))[]
  F = setproperties(F; overlap=n / sqrt(n))
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
  extracter_kwargs=(;),
  updater_kwargs=(;),
  inserter_kwargs=(;),
  normalize=true,
  kws...,
)
  bpc = itn.BeliefPropagationCache(overlap_network, args...)
  ket_vertices = itn.ket_vertices(overlap_network)
  ket_graph = first(
    itn.induced_subgraph(
      itn.underlying_graph(overlap_network), itn.ket_vertices(overlap_network)
    ),
  )
  init_prob = FittingProblem(;
    ket_graph=ket_graph, state=bpc, gauge_region=collect(vertices(ket_graph))
  )

  inserter_kwargs = (; inserter_kwargs..., normalize, set_orthogonal_region=false)
  common_sweep_kwargs = (; nsites, outputlevel, updater_kwargs, inserter_kwargs)
  kwargs_array = [(; common_sweep_kwargs..., sweep=s) for s in 1:nsweeps]
  sweep_iter = sweep_iterator(init_prob, kwargs_array)
  converged_prob = sweep_solve(sweep_iter; outputlevel, kws...)
  return itn.rename_vertices(itn.inv_vertex_map(overlap_network), ket(converged_prob))
end

function fit_tensornetwork(tn, init_state, args...; kwargs...)
  return fit_tensornetwork(itn.inner_network(tn, init_state), args; kwargs...)
end

function itn.truncate(tn; maxdim=default_maxdim(), cutoff=default_cutoff(), kwargs...)
  init_state = itn.ITensorNetwork(
    v -> inds -> it.delta(inds), itn.siteinds(tn); link_space=maxdim
  )
  overlap_network = itn.inner_network(tn, init_state)
  inserter_kwargs = (; trunc=(; cutoff, maxdim))
  return fit_tensornetwork(overlap_network; inserter_kwargs, kwargs...)
end

function itn.apply(
  A::itn.ITensorNetwork,
  x::itn.ITensorNetwork;
  maxdim=default_maxdim(),
  cutoff=default_cutoff(),
  kwargs...,
)
  init_state = itn.ITensorNetwork(
    v -> inds -> it.delta(inds), itn.siteinds(x); link_space=maxdim
  )
  overlap_network = itn.inner_network(x, A, init_state)
  inserter_kwargs = (; trunc=(; cutoff, maxdim))
  return fit_tensornetwork(overlap_network; inserter_kwargs, kwargs...)
end
