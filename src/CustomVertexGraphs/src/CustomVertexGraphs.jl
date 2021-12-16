module CustomVertexGraphs
  using Dictionaries
  using Graphs

  include(joinpath("..", "..", "SubIndexing", "src", "SubIndexing.jl"))
  using .SubIndexing

  export Sub

  include(joinpath("..", "..", "AbstractBijections", "src", "AbstractBijections.jl"))
  using .AbstractBijections

  export set_vertices, CustomVertexEdge

  import Graphs: src, dst, nv, vertices, has_vertex, ne, edges, has_edge, neighbors, outneighbors, inneighbors, all_neighbors, is_directed, add_edge!, add_vertex!, add_vertices!, induced_subgraph, adjacency_matrix, blockdiag, edgetype

  import Base: show, eltype

  struct CustomVertexGraph{V,G<:AbstractGraph,B<:AbstractBijection} <: AbstractGraph{V}
    parent_graph::G
    vertex_to_parent_vertex::B # Invertible map from the vertices to the parent vertices
    function CustomVertexGraph(parent_graph::G, vertex_to_parent_vertex::B) where {G<:AbstractGraph,B<:AbstractBijection}
      @assert issetequal(vertices(parent_graph), image(vertex_to_parent_vertex))
      V = domain_eltype(vertex_to_parent_vertex)
      return new{V,G,B}(parent_graph, vertex_to_parent_vertex)
    end
  end
  vertex_to_parent_vertex(graph::CustomVertexGraph) = graph.vertex_to_parent_vertex

  eltype(g::CustomVertexGraph{V}) where {V} = V

  # Convenient constructor
  set_vertices(graph::AbstractGraph, vertices) = CustomVertexGraph(graph, vertices)

  vertices(graph::CustomVertexGraph) = domain(vertex_to_parent_vertex(graph))

  parent_graph(graph::CustomVertexGraph) = graph.parent_graph
  parent_graph_type(::Type{<:CustomVertexGraph{<:Any,G}}) where {G} = G

  parent_vertices(graph::CustomVertexGraph) = vertices(parent_graph(graph))
  parent_edges(graph::CustomVertexGraph) = edges(parent_graph(graph))
  parent_edgetype(graph::CustomVertexGraph) = edgetype(parent_graph(graph))

  parent_vertex(graph::CustomVertexGraph, vertex) = vertex_to_parent_vertex(graph)[vertex]
  parent_edge(graph::CustomVertexGraph, edge) = parent_edgetype(graph)(parent_vertex(graph, src(edge)), parent_vertex(graph, dst(edge)))
  parent_vertices(graph::CustomVertexGraph, vertices) = [parent_vertex(graph, vertex) for vertex in vertices]
  parent_vertex_to_vertex(graph::CustomVertexGraph, parent_vertex) = vertices(graph)[parent_vertex]

  CustomVertexGraph(vertices::Vector{T}) where T = CustomVertexGraph{Graph{Int}}(vertices)
  CustomVertexDiGraph(vertices::Vector{T}) where T = CustomVertexGraph{DiGraph{Int}}(vertices)

  import Base: Pair, Tuple, show, ==, hash, eltype
  import Graphs: AbstractEdge, src, dst, reverse

  abstract type AbstractCustomVertexEdge{V} <: AbstractEdge{V} end
  struct CustomVertexEdge{V} <: AbstractCustomVertexEdge{V}
    src::V
    dst::V
  end

  CustomVertexEdge{T}(e::CustomVertexEdge{T}) where {T} = e

  CustomVertexEdge(t::Tuple) = CustomVertexEdge(t[1], t[2])
  CustomVertexEdge(p::Pair) = CustomVertexEdge(p.first, p.second)
  CustomVertexEdge{T}(p::Pair) where {T} = CustomVertexEdge(T(p.first), T(p.second))
  CustomVertexEdge{T}(t::Tuple) where {T} = CustomVertexEdge(T(t[1]), T(t[2]))

  eltype(::Type{<:ET}) where ET<:AbstractCustomVertexEdge{T} where T = T

  src(e::AbstractCustomVertexEdge) = e.src
  dst(e::AbstractCustomVertexEdge) = e.dst

  function show(io::IO, mime::MIME"text/plain", e::AbstractCustomVertexEdge)
    show(io, src(e))
    print(io, " => ")
    show(io, dst(e))
    return nothing
  end

  show(io::IO, edge::AbstractCustomVertexEdge) = show(io, MIME"text/plain"(), edge)

  # Conversions
  Pair(e::AbstractCustomVertexEdge) = Pair(src(e), dst(e))
  Tuple(e::AbstractCustomVertexEdge) = (src(e), dst(e))

  CustomVertexEdge{T}(e::AbstractCustomVertexEdge) where T <: Integer = CustomVertexEdge{T}(T(e.src), T(e.dst))

  # Convenience functions
  reverse(e::T) where T<:AbstractCustomVertexEdge = T(dst(e), src(e))
  ==(e1::AbstractCustomVertexEdge, e2::AbstractCustomVertexEdge) = (src(e1) == src(e2) && dst(e1) == dst(e2))
  hash(e::AbstractCustomVertexEdge, h::UInt) = hash(src(e), hash(dst(e), h))

  edgetype(graph::CustomVertexGraph{V}) where {V} = CustomVertexEdge{V}

  default_vertices(graph::AbstractGraph) = Vector(vertices(graph))

  function CustomVertexGraph(graph::AbstractGraph, vertices=default_vertices(graph))
    if length(vertices) != nv(graph)
      throw(ArgumentError("Vertices and parent graph's vertices must have equal length."))
    end
    if !allunique(vertices)
      throw(ArgumentError("Vertices have to be unique."))
    end
    return CustomVertexGraph(graph, inv(Bijection(vertices)))
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
    vertex(parent_vertex) = inv(vertex_to_parent_vertex(graph))[parent_vertex]
    edge(parent_edge) = CustomVertexEdge(vertex(src(parent_edge)), vertex(dst(parent_edge)))
    return map(edge, parent_edges(graph))
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

  function add_edge!(graph::CustomVertexGraph, edge::CustomVertexEdge)
    add_edge!(parent_graph(graph), parent_edge(graph, edge))
    return graph
  end

  function has_edge(graph::CustomVertexGraph, edge::CustomVertexEdge)
    return has_edge(parent_graph(graph), parent_edge(graph, edge))
  end

  # handles single-argument edge constructors such as pairs and tuples
  has_edge(g::CustomVertexGraph, x) = has_edge(g, edgetype(g)(x))
  add_edge!(g::CustomVertexGraph, x) = add_edge!(g, edgetype(g)(x))

  # handles two-argument edge constructors like src,dst
  has_edge(g::CustomVertexGraph, x, y) = has_edge(g, edgetype(g)(x, y))
  add_edge!(g::CustomVertexGraph, x, y) = add_edge!(g, edgetype(g)(x, y))

  function add_vertex!(graph::CustomVertexGraph, v)
    if v ∈ vertices(graph)
      throw(ArgumentError("Duplicate vertices are not allowed"))
    end
    add_vertex!(parent_graph(graph))
    insert!(vertex_to_parent_vertex(graph), v, last(parent_vertices(graph)))
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

  function induced_subgraph(graph::CustomVertexGraph, sub_vertices::Union{Sub,SubIndex,Vector})
    return _induced_subgraph(graph, _get_vertices(graph, sub_vertices))
  end

  function _get_vertices(graph::CustomVertexGraph, sub::Union{Sub,SubIndex})
    return filter(⊆(sub), vertices(graph))
  end

  function _get_vertices(graph::CustomVertexGraph, vertex)
    return vertex
  end

  function _get_vertices(graph::CustomVertexGraph, vertices::Vector)
    return mapreduce(vertex -> _get_vertices(graph, vertex), vcat, vertices)
  end

  function _induced_subgraph(graph::CustomVertexGraph, vertices::Vector)
    sub_graph, _ = induced_subgraph(parent_graph(graph), parent_vertices(graph, vertices))
    return CustomVertexGraph(sub_graph, vertices), vertices
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

  function show(io::IO, mime::MIME"text/plain", graph::CustomVertexGraph)
    println(io, "CustomVertexGraph with $(nv(graph)) vertices:")
    show(io, mime, vertices(graph))
    println(io, "\n")
    println(io, "and $(ne(graph)) edge(s):")
    for e in edges(graph)
      show(io, mime, e)
      println(io)
    end
    return nothing
  end

  show(io::IO, graph::CustomVertexGraph) = show(io, MIME"text/plain"(), graph)
end # module CustomVertexGraphs
