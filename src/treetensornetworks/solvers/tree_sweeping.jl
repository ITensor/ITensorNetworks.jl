direction(step_number) = isodd(step_number) ? Base.Forward : Base.Reverse

function make_region(edge; last_edge=false, nsite=1, region_args=(;), reverse_args=region_args, reverse_step=false)
  if nsite == 1
    site = ([src(edge)], region_args)
    bond = (edge, reverse_args)
    region = reverse_step ? (site, bond) : (site,)
    if last_edge
      return (region..., ([dst(edge)], region_args))
    else
      return region
    end
  elseif nsite == 2
    sites_two = ([src(edge),dst(edge)],region_args)
    sites_one = ([dst(edge)],reverse_args)
    region = reverse_step ? (sites_two, sites_one) : (sites_two,)
    if last_edge
      return (sites_two,)
    else
      return region
    end
  else
    error("nsite=$nsite not supported in alternating_update / update_step")
  end
end

put_kwargs(t::Tuple{Any,NamedTuple}) = t
put_kwargs(t::Tuple{Any,Any,NamedTuple}) = t
put_kwargs(t::Tuple{Any,Any,Any,NamedTuple}) = t

put_kwargs(v::Vector) = (v, (;))
put_kwargs(j::Integer) = ([j], (;))
put_kwargs(t::Tuple{<:Integer}) = (t, (;))
put_kwargs(t::Tuple{<:Integer,<:Integer}) = (t, (;))

function half_sweep(
  dir::Base.ForwardOrdering,
  graph::AbstractGraph,
  region_function;
  root_vertex=default_root_vertex(graph),
  kwargs...,
)
  edges = post_order_dfs_edges(graph, root_vertex)
  V = vertextype(graph)
  E = edgetype(graph)
  steps = []
  for e in edges[1:(end - 1)]
    push!(steps, region_function(e; last_edge=false, kwargs...)...)
  end
  push!(steps, region_function(edges[end]; last_edge=true, kwargs...)...)

  steps = put_kwargs.(steps)

  return steps
end

function half_sweep(dir::Base.ReverseOrdering, args...; kwargs...)
  rev_sweep = []
  for region in reverse(half_sweep(Base.Forward, args...; kwargs...))
    push!(rev_sweep, (reverse(region[1]), region[2:end]...))
  end
  return rev_sweep
end
