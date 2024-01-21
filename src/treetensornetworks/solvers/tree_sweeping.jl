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
