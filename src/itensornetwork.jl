struct Private end

"""
    ITensorNetwork
"""
struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
  data_graph::DataGraph{V,ITensor,ITensor,NamedGraph{V},NamedEdge{V}}
  function ITensorNetwork{V}(::Private, data_graph::DataGraph) where {V}
    return new{V}(data_graph)
  end
end

#
# Data access
#

data_graph(tn::ITensorNetwork) = getfield(tn, :data_graph)
data_graph_type(TN::Type{<:ITensorNetwork}) = fieldtype(TN, :data_graph)

underlying_graph_type(TN::Type{<:ITensorNetwork}) = fieldtype(data_graph_type(TN), :underlying_graph)

ITensorNetwork{V}(data_graph::DataGraph{V}) where {V} = ITensorNetwork{V}(Private(), copy(data_graph))
ITensorNetwork{V}(data_graph::DataGraph) where {V} = ITensorNetwork{V}(Private(), DataGraph{V}(data_graph))

ITensorNetwork(data_graph::DataGraph) = ITensorNetwork{vertextype(data_graph)}(data_graph)

function ITensorNetwork{V}() where {V}
  return ITensorNetwork{V}(data_graph_type(ITensorNetwork{V})())
end

ITensorNetwork() = ITensorNetwork{Any}()

# Conversion
ITensorNetwork(tn::ITensorNetwork) = copy(tn)
ITensorNetwork{V}(tn::ITensorNetwork{V}) where {V} = copy(tn)
function ITensorNetwork{V}(tn::AbstractITensorNetwork) where {V}
  return ITensorNetwork{V}(Private(), DataGraph{V}(data_graph(tn)))
end
ITensorNetwork(tn::AbstractITensorNetwork) = ITensorNetwork{vertextype(tn)}(tn)

convert_vertextype(::Type{V}, tn::ITensorNetwork{V}) where {V} = tn
convert_vertextype(V::Type, tn::ITensorNetwork) = ITensorNetwork{V}(tn)

copy(tn::ITensorNetwork) = ITensorNetwork(copy(data_graph(tn)))

#
# Construction from collections of ITensors
#

function ITensorNetwork(ts::Vector{ITensor})
  return ITensorNetwork{Int}(ts)
end

function ITensorNetwork{V}(ts::Vector{ITensor}) where {V}
  g = NamedGraph{V}(collect(eachindex(ts)))
  tn = ITensorNetwork(g)
  for v in vertices(g)
    tn[v] = ts[v]
  end
  return tn
end

#
# Construction from Graphs
#

function ITensorNetwork{V}(g::AbstractNamedGraph; kwargs...) where {V}
  return ITensorNetwork{V}(IndsNetwork{V}(g; kwargs...))
end

function ITensorNetwork(graph::AbstractNamedGraph; kwargs...)
  return ITensorNetwork{vertextype(graph)}(graph; kwargs...)
end

function ITensorNetwork(g::Graphs.SimpleGraphs.AbstractSimpleGraph; kwargs...)
  return ITensorNetwork(IndsNetwork(g; kwargs...))
end

#
# Construction from IndsNetwork
#

function ITensorNetwork{V}(inds_network::IndsNetwork; kwargs...) where {V}
  # Graphs.jl uses `zero` to create a graph of the same type
  # without any vertices or edges.
  inds_network_merge = typeof(inds_network)(underlying_graph(inds_network); kwargs...)
  inds_network = union(inds_network, inds_network_merge)
  tn = ITensorNetwork{V}()
  for v in vertices(inds_network)
    add_vertex!(tn, v)
  end
  for e in edges(inds_network)
    add_edge!(tn, e)
  end
  for v in vertices(tn)
    siteinds = get(inds_network, v, indtype(inds_network)[])
    linkinds = [get(inds_network, edgetype(inds_network)(v, nv), indtype(inds_network)[]) for nv in neighbors(inds_network, v)]
    setindex_preserve_graph!(tn, ITensor(siteinds, linkinds...), v)
  end
  return tn
end

function ITensorNetwork(inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork{vertextype(inds_network)}(inds_network; kwargs...)
end

function ITensorNetwork(is::IndsNetwork, initstate::Function)
  ψ = ITensorNetwork(is)
  for v in vertices(ψ)
    ψ[v] = state(initstate(v), only(is[v]))
  end
  ψ = insert_links(ψ, edges(is))
  return ψ
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
