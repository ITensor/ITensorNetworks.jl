
function one_site_region(edge; last_edge=false)
  if last_edge
    return [src(edge)], [dst(edge)]
  end
  return ([src(edge)],)
end

function two_site_region(edge; last_edge=false)
  return ([src(edge), dst(edge)],)
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
