module CustomVertexGraphs
  using Dictionaries
  using Graphs
  export CustomVertexGraph, CustomVertexDiGraph, CustomVertexEdge

  import Graphs: src, dst, nv, vertices, has_vertex, ne, edges, has_edge, neighbors, outneighbors, inneighbors, all_neighbors, is_directed, add_edge!, add_vertex!, add_vertices!, induced_subgraph, adjacency_matrix, blockdiag

  import Base: show

  struct CustomVertexGraph{L,G<:AbstractGraph,T} <: AbstractGraph{L}
    parent_graph::G
    vertices::Vector{L}
    vertex_to_parent_vertex_map::Dictionary{L,T}
    function CustomVertexGraph{L,G}(parent_graph::AbstractGraph{T}, vertices, vertex_to_parent_vertex_map) where {L,G,T}
      return new{L,G,T}(parent_graph, vertices, vertex_to_parent_vertex_map)
    end
  end
  parent_graph(graph::CustomVertexGraph) = graph.parent_graph
  vertices(g::CustomVertexGraph) = g.vertices
  vertex_to_parent_vertex_map(graph::CustomVertexGraph) = graph.vertex_to_parent_vertex_map

  parent_graph_type(::Type{<:CustomVertexGraph{<:Any,G}}) where {G} = G

  parent_vertices(graph::CustomVertexGraph) = vertices(parent_graph(graph))
  parent_edges(graph::CustomVertexGraph) = edges(parent_graph(graph))
  parent_vertex(graph::CustomVertexGraph, vertex) = vertex_to_parent_vertex_map(graph)[vertex]
  parent_vertices(graph::CustomVertexGraph, vertices) = [parent_vertex(graph, vertex) for vertex in vertices]

  parent_vertex_to_vertex(graph::CustomVertexGraph, parent_vertex) = vertices(graph)[parent_vertex]

  CustomVertexGraph(vertices::Vector{T}) where T = CustomVertexGraph{Graph{Int}}(vertices)
  CustomVertexDiGraph(vertices::Vector{T}) where T = CustomVertexGraph{DiGraph{Int}}(vertices)

  struct CustomVertexEdge{T} <: AbstractEdge{T}
    src::T
    dst::T
  end

  src(e::CustomVertexEdge) = e.src
  dst(e::CustomVertexEdge) = e.dst

  Base.:(==)(e1::CustomVertexEdge, e2::CustomVertexEdge) = src(e1) == src(e2) && dst(e1) == dst(e2)

  default_vertices(graph::AbstractGraph) = Vector(vertices(graph))

  function CustomVertexGraph(graph::AbstractGraph, vertices=default_vertices(graph))
    if length(vertices) != nv(graph)
      throw(ArgumentError("Labels and parent graph's vertices must have equal length."))
    end
    if !allunique(vertices)
      throw(ArgumentError("Labels have to be unique."))
    end
    vertex_to_parent_vertex_map = Dictionary(vertices, CustomVertexGraphs.vertices(graph))
    return CustomVertexGraph{eltype(vertices),typeof(graph)}(graph, vertices, vertex_to_parent_vertex_map)
  end

  function CustomVertexGraph(graph::AbstractGraph, dims::Tuple{Vararg{Integer}})
    return CustomVertexGraph(graph, vec(Tuple.(CartesianIndices(dims))))
  end

  function CustomVertexGraph(dims::Tuple{Vararg{Integer}})
    return CustomVertexGraph(Graph(prod(dims)), vec(Tuple.(CartesianIndices(dims))))
  end

  function CustomVertexGraph{S}(vertices::Vector) where {S<:AbstractGraph}
    return CustomVertexGraph(S(length(vertices)), vertices)
  end

  has_vertex(g::CustomVertexGraph, v) = v in vertices(g)

  function edges(graph::CustomVertexGraph)
    return map(e -> CustomVertexEdge(vertices(graph)[src(e)], vertices(graph)[dst(e)]), parent_edges(graph))
  end

  function has_edge(graph::CustomVertexGraph, s, d)
    return has_vertex(graph, s) && has_vertex(graph, d) &&
      has_edge(parent_graph(graph), parent_vertex(graph, s), parent_vertex(graph, d))
  end

  function has_edge(graph::CustomVertexGraph, e::AbstractEdge)
    return has_edge(graph, src(e), dst(e))
  end

  for f in [:outneighbors, :inneighbors, :all_neighbors, :neighbors]
    @eval begin
      function $f(graph::CustomVertexGraph, v)
        parent_vertices = $f(parent_graph(graph), parent_vertex(graph, v))
        return [parent_vertex_to_vertex(graph, u) for u ∈ parent_vertices]
      end
    end
  end

  # Ambiguity errors with Graphs.jl
  for f in [
    :neighbors, :inneighbors, :outneighbors, :all_neighbors
  ]
    @eval begin
      $f(tn::CustomVertexGraph, vertex::Integer) = $f(parent_graph(tn), vertex)
    end
  end

  function add_edge!(graph::CustomVertexGraph, src, dst)
    add_edge!(parent_graph(graph), parent_vertex(graph, src), parent_vertex(graph, dst))
    return graph
  end

  function add_edge!(graph::CustomVertexGraph, e)
    add_edge!(graph, src(e), dst(e))
    return graph
  end

  function add_vertex!(graph::CustomVertexGraph, v)
    if v ∈ vertices(graph)
      throw(ArgumentError("Duplicate vertices are not allowed"))
    end
    add_vertex!(parent_graph(graph))
    push!(vertices(graph), v)
    push!(vertex_to_parent_vertex_map(graph), v => nv(parent_graph(graph)))
    return graph
  end

  function add_vertices!(graph::CustomVertexGraph, vertices::Vector)
    if any(v ∈ vertices(graph) for v ∈ vertices)
      throw(ArgumentError("Duplicate vertices are not allowed"))
    end
    for vertex in vertices
      add_vertex!(graph, vertex)
    end
    return graph
  end

  function induced_subgraph(graph::CustomVertexGraph, vertices::Vector)
    sub_g, _ = induced_subgraph(parent_graph(graph), [parent_vertex(graph, v) for v in vertices])
    return CustomVertexGraph(sub_g, vertices), vertices
  end

  is_directed(LG::Type{<:CustomVertexGraph}) = is_directed(parent_graph_type(LG))

  function blockdiag(graph1::CustomVertexGraph, graph2::CustomVertexGraph)
    new_parent_graph = blockdiag(parent_graph(graph1), parent_graph(graph2))
    new_vertices = vcat(vertices(graph1), vertices(graph2))
    return CustomVertexGraph(new_parent_graph, new_vertices)
  end

  for f in [:nv, :ne, :adjacency_matrix]
    @eval begin
      $f(graph::CustomVertexGraph, args...) = $f(parent_graph(graph), args...)
    end
  end

  function show(io::IO, mime::MIME"text/plain", e::CustomVertexEdge)
    show(io, src(e))
    print(io, " => ")
    show(io, dst(e))
    return nothing
  end

  function show(io::IO, mime::MIME"text/plain", graph::CustomVertexGraph)
    println(io, "CustomVertexGraph with $(nv(graph)) vertices:")
    println(io, vertices(graph))
    println(io)
    println(io, "and $(ne(graph)) edges:")
    for e in edges(graph)
      show(io, mime, e)
      println(io)
    end
    return nothing
  end
end
