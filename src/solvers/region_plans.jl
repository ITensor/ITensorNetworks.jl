using Graphs: AbstractGraph, AbstractEdge, edges, dst, src, vertices
using NamedGraphs: GraphsExtensions

#function basic_path_regions(g::AbstractGraph; sweep_kwargs...)
#  fwd_sweep = [([src(e), dst(e)], sweep_kwargs) for e in edges(g)]
#  return [fwd_sweep..., reverse(fwd_sweep)...]
#end

function tdvp_regions(
  g::AbstractGraph, time_step; nsites=1, updater_kwargs, sweep_kwargs...
)
  @assert nsites==1
  fwd_up_args = (; time=(time_step / 2), updater_kwargs...)
  rev_up_args = (; time=(-time_step / 2), updater_kwargs...)

  fwd_sweep = []
  for e in edges(g)
    push!(fwd_sweep, ([src(e)], (; updater_kwargs=fwd_up_args, sweep_kwargs...)))
    push!(fwd_sweep, (e, (; updater_kwargs=rev_up_args, sweep_kwargs...)))
  end
  push!(fwd_sweep, ([dst(last(edges(g)))], (; updater_kwargs=fwd_up_args, sweep_kwargs...)))

  # Reverse regions as well as ordering of regions:
  rev_sweep = [(reverse(rk[1]), rk[2]) for rk in reverse(fwd_sweep)]

  return [fwd_sweep..., rev_sweep...]
end

function overlap(ea::AbstractEdge, eb::AbstractEdge)
  return intersect([src(ea), dst(ea)], [src(eb), dst(eb)])
end

function forward_region(edges, which_edge; nsites=1, region_kwargs=(;))
  current_edge = edges[which_edge]
  if nsites == 1
    #handle edge case
    if current_edge == last(edges)
      overlapping_vertex = only(
        union([overlap(e, current_edge) for e in edges[1:(which_edge - 1)]]...)
      )
      nonoverlapping_vertex = only(
        setdiff([src(current_edge), dst(current_edge)], [overlapping_vertex])
      )
      return [
        ([overlapping_vertex], region_kwargs), ([nonoverlapping_vertex], region_kwargs)
      ]
    else
      future_edges = edges[(which_edge + 1):end]
      future_edges = isa(future_edges, AbstractEdge) ? [future_edges] : future_edges
      overlapping_vertex = only(union([overlap(e, current_edge) for e in future_edges]...))
      nonoverlapping_vertex = only(
        setdiff([src(current_edge), dst(current_edge)], [overlapping_vertex])
      )
      return [([nonoverlapping_vertex], region_kwargs)]
    end
  elseif nsites == 2
    return [([src(current_edge), dst(current_edge)], region_kwargs)]
  end
end

function basic_region_plan(
  graph::AbstractGraph;
  nsites,
  root_vertex=GraphsExtensions.default_root_vertex(graph),
  sweep_kwargs...,
)
  edges = GraphsExtensions.post_order_dfs_edges(graph, root_vertex)
  fwd_sweep = [
    forward_region(edges, i; nsites, region_kwargs=sweep_kwargs) for i in 1:length(edges)
  ]
  fwd_sweep = collect(Iterators.flatten(fwd_sweep))
  return [fwd_sweep..., reverse(fwd_sweep)...]
end
