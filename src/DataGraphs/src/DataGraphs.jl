module DataGraphs

  import Base: get, getindex, setindex!, convert, show, isassigned, eltype

  using Dictionaries

  # Dictionaries.jl patch
  convert(::Type{Dictionary{I,T}}, dict::Dictionary{I,T}) where {I, T} = dict

  using Graphs

  import Graphs: edgetype, ne, nv, vertices, edges, has_edge, has_vertex, neighbors

  export DataGraph

  abstract type AbstractDataGraph{VD,ED,V,E} <: AbstractGraph{V} end

  # Field access
  parent_graph(graph::AbstractDataGraph) = getfield(graph, :parent_graph)
  vertex_data(graph::AbstractDataGraph) = getfield(graph, :vertex_data)
  edge_data(graph::AbstractDataGraph) = getfield(graph, :edge_data)

  # Graphs overloads
  for f in [:edgetype, :nv, :ne, :vertices, :edges, :eltype, :has_edge, :has_vertex, :neighbors]
    @eval begin
      $f(graph::AbstractDataGraph, args...) = $f(parent_graph(graph), args...)
    end
  end

  # Vertex or Edge trait
  abstract type VertexOrEdge end
  struct IsVertex <: VertexOrEdge end
  struct IsEdge <: VertexOrEdge end

  is_vertex_or_edge(graph::AbstractGraph, args...) = error("$args don't represent a vertex or an edge for graph:\n$graph.")
  is_vertex_or_edge(graph::AbstractGraph{V}, I::V) where {V} = IsVertex()
  is_vertex_or_edge(graph::AbstractGraph, I::AbstractEdge) = IsEdge()
  is_vertex_or_edge(graph::AbstractGraph{V}, I::Pair{V,V}) where {V} = IsEdge()
  is_vertex_or_edge(graph::AbstractGraph{V}, I::Tuple{V,V}) where {V} = IsEdge()

  data(::IsVertex, graph::AbstractDataGraph) = vertex_data(graph)
  data(::IsEdge, graph::AbstractDataGraph) = edge_data(graph)
  index_type(::IsVertex, graph::AbstractDataGraph, args...) = eltype(graph)(args...)
  index_type(::IsEdge, graph::AbstractDataGraph, args...) = edgetype(graph)(args...)

  for f in [:getindex, :isassigned]
    @eval begin
      $f(ve::VertexOrEdge, graph::AbstractDataGraph, args...) = $f(data(ve, graph), index_type(ve, graph, args...))
      $f(graph::AbstractDataGraph, args...) = $f(is_vertex_or_edge(graph, args...), graph, args...)
    end
  end

  setindex!(graph::AbstractDataGraph, x, args...) = setindex!(is_vertex_or_edge(graph, args...), graph, x, args...)
  setindex!(ve::VertexOrEdge, graph::AbstractDataGraph, x, args...) = setindex!(data(ve, graph), x, index_type(ve, graph, args...))

  #
  # Helper functions for constructing AbstractDataGraph
  #

  function default_data(
    index_type::Type,
    data_type::Type,
    indices_function::Function,
    parent_graph::AbstractGraph,
    ::Nothing
  )
    return similar(Indices(indices_function(parent_graph)), data_type)
  end

  # Custom dictionary-like constructor that only accepts Pair lists
  function data_dict(
    index_type::Type,
    data_type::Type,
    indices_function::Function,
    parent_graph::AbstractGraph,
    data::Vector{<:Pair}
  )
    indices = index_type.(first.(data))
    values = convert(Vector{data_type}, last.(data))
    return Dictionary{index_type,data_type}(indices, values)
  end

  # Custom dictionary-like constructor that only accepts Pair lists
  function data_dict(
    index_type::Type,
    data_type::Type,
    indices_function::Function,
    parent_graph::AbstractGraph,
    data::Vector
  )
    indices = indices_function(parent_graph)
    values = convert(Vector{data_type}, data)
    return Dictionary{index_type,data_type}(indices, values)
  end

  function default_data(
    index_type::Type,
    data_type::Type,
    indices_function::Function,
    parent_graph::AbstractGraph,
    init_data
  )
    data = data_dict(
      index_type,
      data_type,
      indices_function,
      parent_graph,
      init_data
    )
    return default_data(
      index_type,
      data_type,
      indices_function,
      parent_graph,
      data
    )
  end

  function default_data(
    index_type::Type,
    data_type::Type,
    indices_function::Function,
    parent_graph::AbstractGraph,
    init_data::Dictionary
  )
    data = default_data(
      index_type,
      data_type,
      indices_function,
      parent_graph,
      nothing
    )
    for i in eachindex(init_data)
      if isassigned(init_data, i)
        data[i] = init_data[i]
      end
    end
    return data
  end

  data_type(::Nothing) = Any
  data_type(::Vector{T}) where {T} = T
  data_type(::Vector{Pair{S,T}}) where {S,T} = T

  #
  # Printing
  #

  function show(io::IO, mime::MIME"text/plain", graph::AbstractDataGraph)
    println(io, "DataGraph with $(nv(graph)) vertices:")
    show(io, mime, vertices(graph))
    println(io, "\n")
    println(io, "and $(ne(graph)) edge(s):")
    for e in edges(graph)
      show(io, mime, e)
      println(io)
    end
    println(io)
    println(io, "and vertex data:")
    show(io, mime, vertex_data(graph))
    println(io)
    println(io)
    println(io, "and edge data:")
    show(io, mime, edge_data(graph))
    return nothing
  end

  show(io::IO, graph::AbstractDataGraph) = show(io, MIME"text/plain"(), graph)

  #
  # DataGraph
  #

  struct DataGraph{VD,ED,V,E,G<:AbstractGraph} <: AbstractDataGraph{VD,ED,V,E}
    parent_graph::G
    vertex_data::Dictionary{V,VD}
    edge_data::Dictionary{E,ED}
  end

  function DataGraph{VD,ED}(
    parent_graph::G,
    vertex_data,
    edge_data
  ) where {VD,ED,G<:AbstractGraph}
    V = eltype(parent_graph)
    E = edgetype(parent_graph)
    vertex_data = default_data(V, VD, vertices, parent_graph, vertex_data)
    edge_data = default_data(E, ED, edges, parent_graph, edge_data)
    return DataGraph{VD,ED,V,E,G}(parent_graph, vertex_data, edge_data)
  end

  function DataGraph{VD}(
    parent_graph::AbstractGraph,
    vertex_data,
    edge_data
  ) where {VD}
    ED = data_type(edge_data)
    return DataGraph{VD,ED}(parent_graph, vertex_data, edge_data)
  end

  function (DataGraph{VD,ED} where {VD})(
    parent_graph::AbstractGraph,
    vertex_data,
    edge_data
  ) where {ED}
    VD = data_type(vertex_data)
    return DataGraph{VD,ED}(parent_graph, vertex_data, edge_data)
  end

  function DataGraph(
    parent_graph::AbstractGraph,
    vertex_data,
    edge_data
  )
    VD = data_type(vertex_data)
    ED = data_type(edge_data)
    return DataGraph{VD,ED}(parent_graph, vertex_data, edge_data)
  end

  #
  # kwarg versions call arg versions
  #

  function DataGraph{VD,ED}(
    parent_graph::AbstractGraph;
    vertex_data=nothing,
    edge_data=nothing
  ) where {VD,ED}
    return DataGraph{VD,ED}(parent_graph, vertex_data, edge_data)
  end

  function DataGraph{VD}(
    parent_graph::AbstractGraph;
    vertex_data=nothing,
    edge_data=nothing
  ) where {VD}
    return DataGraph{VD}(parent_graph, vertex_data, edge_data)
  end

  function (DataGraph{VD,ED} where {VD})(
    parent_graph::AbstractGraph;
    vertex_data=nothing,
    edge_data=nothing
  ) where {ED}
    return (DataGraph{VD,ED} where {VD})(parent_graph, vertex_data, edge_data)
  end

  function DataGraph(
    parent_graph::AbstractGraph;
    vertex_data=nothing,
    edge_data=nothing
  )
    return DataGraph(parent_graph, vertex_data, edge_data)
  end

end # module DataGraphs
