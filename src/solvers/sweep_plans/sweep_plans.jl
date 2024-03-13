direction(step_number) = isodd(step_number) ? Base.Forward : Base.Reverse

function overlap(edge_a::AbstractEdge, edge_b::AbstractEdge)
  return intersect(support(edge_a), support(edge_b))
end

function support(edge::AbstractEdge)
  return [src(edge), dst(edge)]
end

support(r) = r

function reverse_region(edges, which_edge; nsites=1, region_args=(;))
  current_edge = edges[which_edge]
  if nsites == 1
    return [(current_edge, region_args)]
  elseif nsites == 2
    if last(edges) == current_edge
      return ()
    end
    future_edges = edges[(which_edge + 1):end]
    future_edges = isa(future_edges, AbstractEdge) ? [future_edges] : future_edges
    #error if more than single vertex overlap
    overlapping_vertex = only(union([overlap(e, current_edge) for e in future_edges]...))
    return [([overlapping_vertex], region_args)]
  end
end

#ToDo: Fix the logic here, currently broken for trees
#Similar to current_ortho, we need to look forward to the next overlapping region
#(which is not necessarily the next region)
function insert_region_intersections(steps, graph; region_args=(;))
  regions = first.(steps)
  intersecting_steps = Any[]
  for i in eachindex(regions)
    i == length(regions) && continue
    region = regions[i]
    intersecting_region = intersect(support(regions[i]), support(regions[i + 1]))
    if isempty(intersecting_region)
      intersecting_region = NamedGraphs.NamedEdge(only(regions[i]), only(regions[i + 1]))
      if !has_edge(graph, intersecting_region)
        error("Edge not in graph")
      end
    end
    push!(intersecting_steps, (intersecting_region, region_args))
  end
  return interleave(steps, intersecting_steps)
end

function forward_region(edges, which_edge; nsites=1, region_args=(;))
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
      return [([overlapping_vertex], region_args), ([nonoverlapping_vertex], region_args)]
    else
      future_edges = edges[(which_edge + 1):end]
      future_edges = isa(future_edges, AbstractEdge) ? [future_edges] : future_edges
      overlapping_vertex = only(union([overlap(e, current_edge) for e in future_edges]...))
      nonoverlapping_vertex = only(
        setdiff([src(current_edge), dst(current_edge)], [overlapping_vertex])
      )
      return [([nonoverlapping_vertex], region_args)]
    end
  elseif nsites == 2
    current_edge = edges[which_edge]
    return [([src(current_edge), dst(current_edge)], region_args)]
  end
end

#
# Helper functions to take a tuple like ([1],[2])
# and append an empty named tuple to it, giving ([1],[2],(;))
#
prepend_missing_namedtuple(t::Tuple) = ((;), t...)
prepend_missing_namedtuple(t::Tuple{<:NamedTuple,Vararg}) = t
function append_missing_namedtuple(t::Tuple)
  return reverse(prepend_missing_namedtuple(reverse(t)))
end

function forward_sweep(
  dir::Base.ForwardOrdering,
  graph::AbstractGraph;
  root_vertex=default_root_vertex(graph),
  region_args,
  reverse_args,
  reverse_step,
  kwargs...,
)
  edges = post_order_dfs_edges(graph, root_vertex)
  regions = collect(
    flatten(map(i -> forward_region(edges, i; region_args, kwargs...), eachindex(edges)))
  )

  if reverse_step
    reverse_regions = collect(
      flatten(
        map(
          i -> reverse_region(edges, i; region_args=reverse_args, kwargs...),
          eachindex(edges),
        ),
      ),
    )
    _check_reverse_sweeps(regions, reverse_regions, graph; kwargs...)
    regions = interleave(regions, reverse_regions)
    #println("insert regions")
    #regions=insert_region_intersections(regions,graph;region_args=reverse_args)
  end
  # Append empty namedtuple to each element if not already present
  # ToDo: Probably not necessary anymore, remove?
  regions = append_missing_namedtuple.(to_tuple.(regions))
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
  kwargs...
)
  extracter, updater, inserter, transform_operator = extend.((extracter, updater, inserter, transform_operator), nsweeps)
  inserter_kwargs, updater_kwargs, extracter_kwargs, transform_operator_kwargs, kwargs =
    expand.((inserter_kwargs, updater_kwargs, extracter_kwargs, transform_operator_kwargs, NamedTuple(kwargs)), nsweeps)
  sweep_plans = []
  for i in 1:nsweeps
    sweep_plan = sweep_plan_func(
      init_state;
      root_vertex,
      pre_region_args=(;
        insert=(inserter[i], inserter_kwargs[i]),
        update=(updater[i], updater_kwargs[i]),
        extract=(extracter[i], extracter_kwargs[i]),
        transform_operator=(transform_operator[i],transform_operator_kwargs[i])
      ),
      kwargs[i]...
    )
    push!(sweep_plans, sweep_plan)
  end
  return sweep_plans
end

function default_sweep_plan(
  graph::AbstractGraph;
  root_vertex=default_root_vertex(graph),
  pre_region_args,
  nsites::Int,
  reverse_step=false,
)
  return vcat(
  [
    forward_sweep(
      direction(half),
      graph;
      root_vertex,
      nsites,
      region_args=(; internal_kwargs=(; half), pre_region_args...),
      reverse_args=region_args,
      reverse_step,
    ) for half in 1:2
  ]...,
  )
end

function tdvp_sweep_plan(
  graph::AbstractGraph;
  root_vertex=default_root_vertex(graph),
  pre_region_args,
  reverse_step=true,
  order::Int,
  nsites::Int,
  time_step::Number,
)
  sweep_plan = []
  for (substep, fac) in enumerate(sub_time_steps(order))
    sub_time_step = time_step * fac
    append!(sweep_plan,
      forward_sweep(direction(substep), graph;
      root_vertex,
      nsites,
      region_args=(;
        internal_kwargs=(; substep, time_step=sub_time_step), pre_region_args...
      ),
      reverse_args=(;
        internal_kwargs=(; substep, time_step=-sub_time_step), pre_region_args...
      ),
      reverse_step,
      )
    )
  end
  return sweep_plan
end



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
