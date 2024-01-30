direction(step_number) = isodd(step_number) ? Base.Forward : Base.Reverse

function make_region(
  edge;
  last_edge=false,
  nsites=1,
  region_args=(;),
  reverse_args=region_args,
  reverse_step=false,
)
  if nsites == 1
    site = ([src(edge)], region_args)
    bond = (edge, reverse_args)
    region = reverse_step ? (site, bond) : (site,)
    if last_edge
      return (region..., ([dst(edge)], region_args))
    else
      return region
    end
  elseif nsites == 2
    sites_two = ([src(edge), dst(edge)], region_args)
    sites_one = ([dst(edge)], reverse_args)
    region = reverse_step ? (sites_two, sites_one) : (sites_two,)
    if last_edge
      return (sites_two,)
    else
      return region
    end
  else
    error("nsites=$nsites not supported in alternating_update / update_step")
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

function half_sweep(
  dir::Base.ForwardOrdering,
  graph::AbstractGraph,
  region_function;
  root_vertex=default_root_vertex(graph),
  kwargs...,
)
  edges = post_order_dfs_edges(graph, root_vertex)
  steps = collect(
    flatten(map(e -> region_function(e; last_edge=(e == edges[end]), kwargs...), edges))
  )
  # Append empty namedtuple to each element if not already present
  steps = append_missing_namedtuple.(to_tuple.(steps))
  return steps
end

function half_sweep(dir::Base.ReverseOrdering, args...; kwargs...)
  return map(
    region -> (reverse(region[1]), region[2:end]...),
    reverse(half_sweep(Base.Forward, args...; kwargs...)),
  )
end


function default_sweep_plan(nsites, graph::AbstractGraph; kwargs...)  ###move this to a different file, algorithmic level idea
  return vcat(
    [
      half_sweep(
        direction(half),
        graph,
        make_region;
        nsites,
        region_args=(; half_sweep=half),
        kwargs...,
      ) for half in 1:2
    ]...,
  )
end


function tdvp_sweep_plan(
  order::Int,
  nsites::Int,
  time_step::Number,
  graph::AbstractGraph;
  root_vertex=default_root_vertex(graph),
  reverse_step=true,
)
  sweep_plan = []
  for (substep, fac) in enumerate(sub_time_steps(order))
    sub_time_step = time_step * fac
    half = half_sweep(
      direction(substep),
      graph,
      make_region;
      root_vertex,
      nsites,
      region_args=(; substep, time_step=sub_time_step),
      reverse_args=(; substep, time_step=-sub_time_step),
      reverse_step,
    )
    append!(sweep_plan, half)
  end
  return sweep_plan
end


function _extend_sweeps_param(param, nsweeps)
  if param isa Number
    eparam = fill(param, nsweeps)
  else
    length(param) >= nsweeps && return param[1:nsweeps]
    eparam = Vector(undef, nsweeps)
    eparam[1:length(param)] = param
    eparam[(length(param) + 1):end] .= param[end]
  end
  return eparam
end

function process_sweeps(
  nsweeps;
  cutoff=fill(1E-16, nsweeps),
  maxdim=fill(typemax(Int), nsweeps),
  mindim=fill(1, nsweeps),
  noise=fill(0.0, nsweeps),
  kwargs...,
)
  maxdim = _extend_sweeps_param(maxdim, nsweeps)
  mindim = _extend_sweeps_param(mindim, nsweeps)
  cutoff = _extend_sweeps_param(cutoff, nsweeps)
  noise = _extend_sweeps_param(noise, nsweeps)
  return maxdim, mindim, cutoff, noise, kwargs
end

function sweep_printer(; outputlevel, state, which_sweep, sw_time)
  if outputlevel >= 1
    print("After sweep ", which_sweep, ":")
    print(" maxlinkdim=", maxlinkdim(state))
    print(" cpu_time=", round(sw_time; digits=3))
    println()
    flush(stdout)
  end
end