#module SimpleMetaGraphs
  using Dictionaries
  using Graphs

  import Base: get, getindex, setindex!

  import Graphs: edgetype, ne, nv

  abstract type AbstractDataGraph{VD,ED,V,E} <: AbstractGraph{V} end

  # Field access
  parent_graph(graph::AbstractDataGraph) = getfield(graph, :parent_graph)
  vertex_data(graph::AbstractDataGraph) = getfield(graph, :vertex_data)
  edge_data(graph::AbstractDataGraph) = getfield(graph, :edge_data)

  # Graphs overloads
  for f in [:edgetype, :nv, :ne]
    @eval begin
      $f(graph::AbstractDataGraph, args...) = $f(parent_graph(graph), args...)
    end
  end

  getindex(graph::AbstractDataGraph{V}, v::V) where {V} = vertex_data(graph)[v]
  getindex(graph::AbstractDataGraph, e::AbstractEdge) = edge_data(graph)[e]
  getindex(graph::AbstractDataGraph, e::Pair) = graph[edgetype(graph)(e)]
  getindex(graph::AbstractDataGraph{V}, src::V, dst::V) where {V} = graph[edgetype(graph)(src, dst)]

  setindex!(graph::AbstractDataGraph{V}, x, v::V) where {V} = (vertex_data(graph)[v] = x; return graph)
  setindex!(graph::AbstractDataGraph, x, e::AbstractEdge) = (edge_data(graph)[e] = x; return graph)
  setindex!(graph::AbstractDataGraph, x, e::Pair) = (graph[edgetype(graph)(e)] = x; return graph)
  setindex!(graph::AbstractDataGraph{V}, x, src::V, dst::V) where {V} = (graph[edgetype(graph)(src, dst)] = x; return graph)

  #
  # DataGraph
  #

  struct DataGraph{VD,ED,V,E,G<:AbstractGraph} <: AbstractDataGraph{VD,ED,V,E}
    parent_graph::G
    vertex_data::Dictionary{V,VD}
    edge_data::Dictionary{E,ED}
    function DataGraph{VD,ED}(graph::G) where {VD,ED,G<:AbstractGraph}
      V = eltype(graph)
      E = edgetype(graph)
      vertex_data = similar(Indices(vertices(graph)), Vector{VD})
      edge_data = similar(Indices(edges(graph)), Vector{ED})
      display(vertex_data)
      display(edge_data)
      return new{VD,ED,V,E,G}(graph, vertex_data, edge_data)
    end
  end

  #function AbstractDataGraph(
  #  parent_graph::AbstractGraph,
  #  vertex_data::AbstractDictionary,
  #  edge_data::AbstractDictionary)
  #  V = eltype(parent_graph)
  #  VD = eltype(vertex_data)
  #  E = edgetype(parent_graph)
  #  ED = eltype(edge_data)
  #  G = typeof(parent_graph)
  #  return new{V,VD,E,ED,G}(parent_graph, vertex_data, edge_metadata)
  #end

  #function DataGraph(graph::AbstractGraph; VertexMetaType=Any, EdgeMetaType=Any)
  #  # Make a MetaGraph
  #  vertex_indices = Indices(vertices(graph))
  #  edge_indices = Indices(edges(graph))
  #end

#end # module DataGraphs
