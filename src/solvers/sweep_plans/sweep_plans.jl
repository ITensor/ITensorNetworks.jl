using Graphs: AbstractEdge, dst, src, edges
using NamedGraphs.GraphsExtensions: GraphsExtensions

direction(step_number) = isodd(step_number) ? Base.Forward : Base.Reverse

function overlap(edge_a::AbstractEdge, edge_b::AbstractEdge)
  return intersect(support(edge_a), support(edge_b))
end

function support(edge::AbstractEdge)
  return [src(edge), dst(edge)]
end

support(r) = r

function reverse_region(edges, which_edge; nsites=1, region_kwargs=(;))
  current_edge = edges[which_edge]
  if nsites == 1
    return [(current_edge, region_kwargs)]
  elseif nsites == 2
    if last(edges) == current_edge
      return ()
    end
    future_edges = edges[(which_edge + 1):end]
    future_edges = isa(future_edges, AbstractEdge) ? [future_edges] : future_edges
    #error if more than single vertex overlap
    overlapping_vertex = only(union([overlap(e, current_edge) for e in future_edges]...))
    return [([overlapping_vertex], region_kwargs)]
  end
end

function forward_region(edges, which_edge; nsites=1, region_kwargs=(;))
  if nsites == 1
    current_edge = edges[which_edge]
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
    current_edge = edges[which_edge]
    return [([src(current_edge), dst(current_edge)], region_kwargs)]
  end
end

function forward_sweep(
  dir::Base.ForwardOrdering,
  graph::AbstractGraph;
  root_vertex=GraphsExtensions.default_root_vertex(graph),
  region_kwargs,
  reverse_kwargs=region_kwargs,
  reverse_step=false,
  kwargs...,
)
  edges = post_order_dfs_edges(graph, root_vertex)
  regions = collect(
    flatten(map(i -> forward_region(edges, i; region_kwargs, kwargs...), eachindex(edges)))
  )

  if reverse_step
    reverse_regions = collect(
      flatten(
        map(
          i -> reverse_region(edges, i; region_kwargs=reverse_kwargs, kwargs...),
          eachindex(edges),
        ),
      ),
    )
    _check_reverse_sweeps(regions, reverse_regions, graph; kwargs...)
    regions = interleave(regions, reverse_regions)
  end

  return regions
end

#ToDo: is there a better name for this? unidirectional_sweep? traversal?
function forward_sweep(dir::Base.ReverseOrdering, args...; kwargs...)
  return reverse(forward_sweep(Base.Forward, args...; kwargs...))
end

function default_sweep_plans(
  nsweeps,
  init_state;
  sweep_plan_func=default_sweep_plan,
  root_vertex,
  extracter,
  extracter_kwargs,
  updater,
  updater_kwargs,
  inserter,
  inserter_kwargs,
  transform_operator,
  transform_operator_kwargs,
  kwargs...,
)
  extracter, updater, inserter, transform_operator =
    extend_or_truncate.((extracter, updater, inserter, transform_operator), nsweeps)
  inserter_kwargs, updater_kwargs, extracter_kwargs, transform_operator_kwargs, kwargs =
    expand.(
      (
        inserter_kwargs,
        updater_kwargs,
        extracter_kwargs,
        transform_operator_kwargs,
        NamedTuple(kwargs),
      ),
      nsweeps,
    )
  sweep_plans = []
  for i in 1:nsweeps
    sweep_plan = sweep_plan_func(
      init_state;
      root_vertex,
      region_kwargs=(;
        inserter=inserter[i],
        inserter_kwargs=inserter_kwargs[i],
        updater=updater[i],
        updater_kwargs=updater_kwargs[i],
        extracter=extracter[i],
        extracter_kwargs=extracter_kwargs[i],
        transform_operator=transform_operator[i],
        transform_operator_kwargs=transform_operator_kwargs[i],
      ),
      kwargs[i]...,
    )
    push!(sweep_plans, sweep_plan)
  end
  return sweep_plans
end

function bp_sweep_plan(
  g::AbstractGraph;
  root_vertex=GraphsExtensions.default_root_vertex(graph),
  region_kwargs,
  nsites::Int,
  es=vcat(collect(edges(g)), collect(reverse(edges(g)))),
  vs=vcat(reverse(collect(vertices(g))), collect(vertices(g))),
)
  region_kwargs = (; internal_kwargs=(;), region_kwargs...)

  if nsites == 2
    return collect(flatten(map(e -> [([src(e), dst(e)], region_kwargs)], es)))
  elseif nsites == 1
    return collect(flatten(map(v -> [([v], region_kwargs)], vs)))
  end
end

function default_sweep_plan(
  graph::AbstractGraph;
  root_vertex=GraphsExtensions.default_root_vertex(graph),
  region_kwargs,
  nsites::Int,
)
  return vcat(
    [
      forward_sweep(
        direction(half),
        graph;
        root_vertex,
        nsites,
        region_kwargs=(; internal_kwargs=(; half), region_kwargs...),
      ) for half in 1:2
    ]...,
  )
end

function tdvp_sweep_plan(
  graph::AbstractGraph;
  root_vertex=GraphsExtensions.default_root_vertex(graph),
  region_kwargs,
  reverse_step=true,
  order::Int,
  nsites::Int,
  time_step::Number,
  t_evolved::Number,
)
  sweep_plan = []
  for (substep, fac) in enumerate(sub_time_steps(order))
    sub_time_step = time_step * fac
    append!(
      sweep_plan,
      forward_sweep(
        direction(substep),
        graph;
        root_vertex,
        nsites,
        region_kwargs=(;
          internal_kwargs=(; substep, time_step=sub_time_step, t=t_evolved),
          region_kwargs...,
        ),
        reverse_kwargs=(;
          internal_kwargs=(; substep, time_step=-sub_time_step, t=t_evolved),
          region_kwargs...,
        ),
        reverse_step,
      ),
    )
  end
  return sweep_plan
end

#ToDo: Move to test.
function _check_reverse_sweeps(forward_sweep, reverse_sweep, graph; nsites, kwargs...)
  fw_regions = first.(forward_sweep)
  bw_regions = first.(reverse_sweep)
  if nsites == 2
    fw_verts = flatten(fw_regions)
    bw_verts = flatten(bw_regions)
    for v in vertices(graph)
      @assert isone(count(isequal(v), fw_verts) - count(isequal(v), bw_verts))
    end
  elseif nsites == 1
    fw_verts = flatten(fw_regions)
    bw_edges = bw_regions
    for v in vertices(graph)
      @assert isone(count(isequal(v), fw_verts))
    end
    for e in edges(graph)
      @assert isone(count(x -> (isequal(x, e) || isequal(x, reverse(e))), bw_edges))
    end
  end
  return true
end
