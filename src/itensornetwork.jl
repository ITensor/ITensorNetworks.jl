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

function ITensorNetwork(ts::Vector{ITensor})
  g = NamedDimGraph(ts)
  tn = ITensorNetwork(g)
  for v in eachindex(ts)
    tn[v] = ts[v]
  end
  return tn
end

#
# Construction from Graphs
#

function _ITensorNetwork(g::NamedDimGraph, site_space::Nothing, link_space::Nothing)
  dg = NamedDimDataGraph{ITensor,ITensor}(copy(g))
  return ITensorNetwork(dg)
end

function ITensorNetwork(g::NamedDimGraph; kwargs...)
  return ITensorNetwork(IndsNetwork(g; kwargs...))
end

function ITensorNetwork(g::Graph; kwargs...)
  return ITensorNetwork(IndsNetwork(g; kwargs...))
end

#
# Construction from IndsNetwork
#

# Alternative implementation:
# edge_data(e) = [edge_index(e, link_space)]
# is_assigned = assign_data(is; edge_data)
function _ITensorNetwork(is::IndsNetwork, link_space)
  is_assigned = copy(is)
  for e in edges(is)
    is_assigned[e] = [edge_index(e, link_space)]
  end
  return _ITensorNetwork(is_assigned, nothing)
end

get_assigned(d, i, default) = isassigned(d, i) ? d[i] : default

function _ITensorNetwork(is::IndsNetwork, link_space::Nothing)
  g = underlying_graph(is)
  tn = _ITensorNetwork(g, nothing, nothing)
  for v in vertices(tn)
    siteinds = get_assigned(is, v, Index[])
    linkinds = [get_assigned(is, v => nv, Index[]) for nv in neighbors(is, v)]
    setindex_preserve_graph!(tn, ITensor(siteinds, linkinds...), v)
  end
  return tn
end

function ITensorNetwork(is::IndsNetwork; link_space=nothing)
  return _ITensorNetwork(is, link_space)
end

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

function ITensorNetwork(is::IndsNetwork, initstate::Function)
  ψ = ITensorNetwork(is)
  for v in vertices(ψ)
    ψ[v] = state(initstate(v), only(is[v]))
  end
  ψ = insert_links(ψ, edges(is))
  return ψ
end
