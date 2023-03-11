# 
# Sweep step
# 

"""
  struct SweepStep{V}

Auxiliary object specifying a single local update step in a tree sweeping algorithm.
"""
struct SweepStep{V} # TODO: parametrize on position type instead of vertex type
  pos::Union{Vector{<:V},NamedEdge{V}}
  data::NamedTuple
end

# field access
pos(st::SweepStep) = st.pos
nsite(st::SweepStep) = isa(pos(st), AbstractEdge) ? 0 : length(pos(st))
data(st::SweepStep) = st.data

# utility
current_ortho(st::SweepStep) = current_ortho(typeof(pos(st)), st)
current_ortho(::Type{<:Vector{<:V}}, st::SweepStep{V}) where {V} = first(pos(st)) # not very clean...
current_ortho(::Type{NamedEdge{V}}, st::SweepStep{V}) where {V} = src(pos(st))

# reverse
Base.reverse(s::SweepStep{V}) where {V} = SweepStep{V}(reverse(pos(s)), data(s))

function Base.:(==)(s1::SweepStep{V}, s2::SweepStep{V}) where {V}
  return pos(s1) == pos(s2) && data(s1) == data(s2)
end

# 
# Pre-defined sweeping paradigms
# 

function one_site_sweep(
  direction::Base.ForwardOrdering,
  graph::AbstractGraph{V},
  reverse_step;
  root_vertex::V=default_root_vertex(graph),
  kwargs...,
) where {V}
  edges = post_order_dfs_edges(graph, root_vertex)
  steps = SweepStep{V}[]
  for e in edges
    push!(steps, SweepStep{V}([src(e)], (;time_direction = +1)))
    reverse_step && push!(steps, SweepStep{V}(e, (;time_direction = -1)))
  end
  push!(steps, SweepStep{V}([root_vertex], (;time_direction = +1)))
  return steps
end

function one_site_sweep(direction::Base.ReverseOrdering, args...; kwargs...)
  return reverse(reverse.(one_site_sweep(Base.Forward, args...; kwargs...)))
end

function two_site_sweep(
  direction::Base.ForwardOrdering,
  graph::AbstractGraph{V},
  reverse_step;
  root_vertex::V=default_root_vertex(graph),
  kwargs...,
) where {V}
  edges = post_order_dfs_edges(graph, root_vertex)
  steps = SweepStep{V}[]
  for e in edges[1:(end - 1)]
    push!(steps, SweepStep{V}([src(e), dst(e)], (;time_direction = +1)))
    reverse_step && push!(steps, SweepStep{V}([dst(e)], (;time_direction = -1)))
  end
  push!(steps, SweepStep{V}([src(edges[end]), dst(edges[end])], (;time_direction = +1)))
  return steps
end

function two_site_sweep(direction::Base.ReverseOrdering, args...; kwargs...)
  return reverse(reverse.(two_site_sweep(Base.Forward, args...; kwargs...)))
end
