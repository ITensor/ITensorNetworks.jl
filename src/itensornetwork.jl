"""
    ITensorNetwork
"""
struct ITensorNetwork <: AbstractITensorNetwork
  data_graph::UniformDataGraph{ITensor}
end

#
# Data access
#

data_graph(tn::ITensorNetwork) = getfield(tn, :data_graph)

copy(tn::ITensorNetwork) = ITensorNetwork(copy(data_graph(tn)))

#
# Construction from collections of ITensors
#

function ITensorNetwork(ts::Vector{<:ITensor}, g::AbstractGraph)
  tn = ITensorNetwork(g)
  for v in eachindex(ts)
    tn[v] = ts[v]
  end
  return tn
end

function ITensorNetwork(ts::Vector{<:ITensor})
  return ITensorNetwork(ts, NamedDimGraph(ts))
end

#
# Construction from Graphs
#

# catch-all for default ElType
function ITensorNetwork(g::AbstractGraph, args...; kwargs...)
  return ITensorNetwork(Float64, g, args...; kwargs...)
end

function _ITensorNetwork(g::NamedDimGraph, site_space::Nothing, link_space::Nothing)
  dg = NamedDimDataGraph{ITensor,ITensor}(copy(g))
  return ITensorNetwork(dg)
end

function ITensorNetwork(::Type{ElT}, g::NamedDimGraph; kwargs...) where {ElT<:Number}
  return ITensorNetwork(ElT, IndsNetwork(g; kwargs...))
end

function ITensorNetwork(::Type{ElT}, g::Graph; kwargs...) where {ElT<:Number}
  return ITensorNetwork(ElT, IndsNetwork(g; kwargs...))
end

#
# Construction from IndsNetwork
#

# Alternative implementation:
# edge_data(e) = [edge_index(e, link_space)]
# is_assigned = assign_data(is; edge_data)
function _ITensorNetwork(::Type{ElT}, is::IndsNetwork, link_space) where {ElT<:Number}
  is_assigned = copy(is)
  for e in edges(is)
    is_assigned[e] = [edge_index(e, link_space)]
  end
  return _ITensorNetwork(ElT, is_assigned, nothing)
end

get_assigned(d, i, default) = isassigned(d, i) ? d[i] : default

function _ITensorNetwork(
  ::Type{ElT}, is::IndsNetwork, link_space::Nothing
) where {ElT<:Number}
  g = underlying_graph(is)
  tn = _ITensorNetwork(g, nothing, nothing)
  for v in vertices(tn)
    siteinds = get_assigned(is, v, Index[])
    linkinds = [get_assigned(is, v => nv, Index[]) for nv in neighbors(is, v)]
    setindex_preserve_graph!(tn, ITensor(ElT, siteinds, linkinds...), v)
  end
  return tn
end

function ITensorNetwork(
  ::Type{ElT}, is::IndsNetwork; link_space=nothing
) where {ElT<:Number}
  return _ITensorNetwork(ElT, is, link_space)
end

#
# Construction from IndsNetwork and state map
#

function insert_links(ψ::ITensorNetwork, edges::Vector=edges(ψ); cutoff=1e-15)
  for e in edges
    # Define this to work?
    # ψ = factorize(ψ, e; cutoff)
    ψᵥ₁, ψᵥ₂ = factorize(ψ[src(e)] * ψ[dst(e)], inds(ψ[src(e)]); cutoff, tags=edge_tag(e))
    ψ[src(e)] = ψᵥ₁
    ψ[dst(e)] = ψᵥ₂
  end
  return ψ
end

function ITensorNetwork(
  ::Type{ElT}, is::IndsNetwork, states_map::Dictionary
) where {ElT<:Number}
  ψ = ITensorNetwork(is)
  for v in vertices(ψ)
    ψ[v] = convert_eltype(ElT, state(only(is[v]), states_map[v]))
  end
  ψ = insert_links(ψ, edges(is))
  return ψ
end

function ITensorNetwork(
  ::Type{ElT}, is::IndsNetwork, state::Union{String,Integer}
) where {ElT<:Number}
  states_map = dictionary([v => state for v in vertices(is)])
  return ITensorNetwork(ElT, is, states_map)
end

function ITensorNetwork(::Type{ElT}, is::IndsNetwork, state::Function) where {ElT<:Number}
  states_map = dictionary([v => state(v) for v in vertices(is)])
  return ITensorNetwork(ElT, is, states_map)
end

#
# Random constructor
#

# TODO: generalize to other random number distributions
function randomITensorNetwork(s; link_space)
  ψ = ITensorNetwork(s; link_space)
  for v in vertices(ψ)
    ψᵥ = copy(ψ[v])
    randn!(ψᵥ)
    ψᵥ ./= norm(ψᵥ)
    ψ[v] = ψᵥ
  end
  return ψ
end
