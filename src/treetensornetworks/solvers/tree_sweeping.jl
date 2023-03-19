
function one_site_half_sweep(
  dir::Base.ForwardOrdering,
  graph::AbstractGraph;
  include_edges=false,
  root_vertex=default_root_vertex(graph),
  kwargs...,
)
  edges = post_order_dfs_edges(graph, root_vertex)
  V = vertextype(graph)
  E = edgetype(graph)
  steps = Union{Vector{<:V},E}[]
  for e in edges
    push!(steps, [src(e)])
    include_edges && push!(steps, e)
  end
  push!(steps, [root_vertex])
  return steps
end

function one_site_half_sweep(dir::Base.ReverseOrdering, args...; kwargs...)
  return reverse(reverse.(one_site_half_sweep(Base.Forward, args...; kwargs...)))
end

function two_site_half_sweep(
  dir::Base.ForwardOrdering,
  graph::AbstractGraph;
  include_edges=false,
  root_vertex=default_root_vertex(graph),
  kwargs...,
)
  edges = post_order_dfs_edges(graph, root_vertex)
  V = vertextype(graph)
  E = edgetype(graph)
  steps = Union{Vector{<:V},E}[]
  for e in edges[1:(end - 1)]
    push!(steps, [src(e), dst(e)])
    include_edges && push!(steps, [dst(e)])
  end
  push!(steps, [src(edges[end]), dst(edges[end])])
  return steps
end

function two_site_half_sweep(dir::Base.ReverseOrdering, args...; kwargs...)
  return reverse(reverse.(two_site_half_sweep(Base.Forward, args...; kwargs...)))
end
