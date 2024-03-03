direction(step_number) = isodd(step_number) ? Base.Forward : Base.Reverse

function interleave(a::Vector, b::Vector)
  ab = flatten(collect(zip(a, b)))
  if length(a) == length(b)
    return ab
  elseif length(a) == length(b) + 1
    return append!(ab, [last(a)])
  else
    error(
      "Trying to interleave vectors of length $(length(a)) and $(length(b)), not implemented.",
    )
  end
end

function overlap(edge_a::AbstractEdge, edge_b::AbstractEdge)
  return intersect([src(edge_a), dst(edge_a)], [src(edge_b), dst(edge_b)])
end

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

function forward_region(edges, which_edge; nsites=1, region_args=(;))
  if nsites == 1
    current_edge = edges[which_edge]
    #handle edge case
    if current_edge == last(edges)
      overlapping_vertex = only(
        union([overlap(e, current_edge) for e in edges[1:(which_edge - 1)]]...)
      ) #union(overlap.(edges[1:(which_edge-1)],current_edge))
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
  forward_steps = collect(
    flatten(map(i -> forward_region(edges, i; region_args, kwargs...), eachindex(edges)))
  )
  if reverse_step
    reverse_steps = collect(
      flatten(
        map(
          i -> reverse_region(edges, i; region_args=reverse_args, kwargs...),
          eachindex(edges),
        ),
      ),
    )
    steps = interleave(forward_steps, reverse_steps)
  else
    steps = forward_steps
  end
  # Append empty namedtuple to each element if not already present
  steps = append_missing_namedtuple.(to_tuple.(steps))
  return steps
end

#ToDo: is there a better name for this? unidirectional_sweep? traversal?
function forward_sweep(dir::Base.ReverseOrdering, args...; kwargs...)
  return map(
    region -> (reverse(region[1]), region[2:end]...),
    reverse(forward_sweep(Base.Forward, args...; kwargs...)),
  )
end

function default_sweep_plan(nsites, graph::AbstractGraph; pre_region_args=(;), kwargs...)  ###move this to a different file, algorithmic level idea
  return vcat(
    [
      half_sweep(
        direction(half),
        graph,
        make_region;
        nsites,
        region_args=(; internal_kwargs=(; half), pre_region_args...),
        reverse_args=region_args,
        kwargs...,
      ) for half in 1:2
    ]...,
  )
end

function tdvp_sweep_plans(
  nsweeps,
  t,
  time_step,
  order,
  nsites,
  init_state;
  root_vertex,
  reverse_step,
  extracter,
  extracter_kwargs,
  updater,
  updater_kwargs,
  inserter,
  inserter_kwargs,
)
  nsweeps, time_step = _compute_nsweeps(nsweeps, t, time_step)
  order, nsites, reverse_step = extend.((order, nsites, reverse_step), nsweeps)
  extracter, updater, inserter = extend.((extracter, updater, inserter), nsweeps)
  inserter_kwargs, updater_kwargs, extracter_kwargs =
    expand.((inserter_kwargs, updater_kwargs, extracter_kwargs), nsweeps)
  sweep_plans = []
  for i in 1:nsweeps
    sweep_plan = tdvp_sweep_plan(
      order[i],
      nsites[i],
      time_step[i],
      init_state;
      root_vertex,
      reverse_step=reverse_step[i],
      pre_region_args=(;
        insert=(inserter[i], inserter_kwargs[i]),
        update=(updater[i], updater_kwargs[i]),
        extract=(extracter[i], extracter_kwargs[i]),
      ),
    )
    #@show sweep_plan
    push!(sweep_plans, sweep_plan)
  end
  return sweep_plans
end

#ToDo: This is currently coupled with the updater signature, which is undesirable.
function tdvp_sweep_plan(
  order::Int,
  nsites::Int,
  time_step::Number,
  graph::AbstractGraph;
  root_vertex=default_root_vertex(graph),
  pre_region_args,
  reverse_step=true,
)
  sweep_plan = []
  for (substep, fac) in enumerate(sub_time_steps(order))
    sub_time_step = time_step * fac
    half = forward_sweep(
      direction(substep),
      graph;
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
    append!(sweep_plan, half)
  end
  return sweep_plan
end

function default_sweep_printer(; outputlevel, state, which_sweep, sweep_time)
  if outputlevel >= 1
    print("After sweep ", which_sweep, ":")
    print(" maxlinkdim=", maxlinkdim(state))
    print(" cpu_time=", round(sweep_time; digits=3))
    println()
    flush(stdout)
  end
end
