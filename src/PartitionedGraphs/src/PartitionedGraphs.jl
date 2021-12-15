module PartitionedGraphs
  using Dictionaries
  using Graphs
  export PartitionedGraph, PartitionedDiGraph, Partition, partitions, set_partitions, subgraph

  import Base: convert, show

  import Graphs: src, dst, nv, vertices, has_vertex, ne, edges, has_edge, neighbors, outneighbors, inneighbors, all_neighbors, is_directed, add_edge!, add_vertex!, add_vertices!, induced_subgraph, adjacency_matrix, blockdiag

  struct Partition{P}
    partition::P
  end
  convert(::Type{Partition{P}}, x) where {P} = Partition{P}(x)
  convert(::Type{Partition{P}}, x::Partition{P}) where {P} = x
  function show(io::IO, ::MIME"text/plain", p::Partition)
    print(io, "Partition(")
    show(io, p.partition)
    print(io, ")")
    return nothing
  end
  show(io::IO, p::Partition) = show(io, MIME"text/plain"(), p)

  struct PartitionedGraph{P,G<:AbstractGraph,T} <: AbstractGraph{T}
    graph::G
    partitions::Dictionary{Partition{P},Vector{T}}
    function PartitionedGraph(graph::AbstractGraph{T}, partitions::Dictionary{Partition{P}}) where {T,P}
      sub_vertices = reduce(vcat, partitions)
      if !issetequal(sub_vertices, vertices(graph))
        error("Partitions $(partitions) must be the same as the vertices of the graph $(collect(vertices(graph)))")
      end
      return new{P,typeof(graph),T}(graph, partitions)
    end
  end
  parent_graph(graph::PartitionedGraph) = graph.graph
  partitions(graph::PartitionedGraph) = graph.partitions

  parent_graph_type(::Type{<:PartitionedGraph{L,G}}) where {L,G} = G

  set_partitions(graph::PartitionedGraph, partitions) = PartitionedGraph(parent_graph(graph), partitions)

  # Version that accepts partitions as a list of pairs, such as
  # ["A" => [1, 2, 3], "B" => [4, 5, 6]]
  function PartitionedGraph(graph::AbstractGraph, partitions)
    new_partitions = Dictionary(first.(partitions), last.(partitions))
    return PartitionedGraph(graph, new_partitions)
  end

  function PartitionedGraph(graph::AbstractGraph, partitions::Dictionary{K,T}) where {K,T}
    new_partitions = Dictionary{Partition{K},T}(partitions)
    return PartitionedGraph(graph, new_partitions)
  end

  function induced_subgraph(graph::PartitionedGraph, vertices::AbstractVector)
    sub_g, vmap = induced_subgraph(parent_graph(graph), vertices)
    parts = partitions(graph)
    sub_parts = Dictionary{keytype(parts),eltype(parts)}()
    for k in keys(parts)
      sub_verts = parts[k] ∩ vertices
      if !isempty(sub_verts)
        insert!(sub_parts, k, sub_verts)
      end
    end
    return PartitionedGraph(sub_g, sub_parts), vmap
  end

  function induced_subgraph(graph::PartitionedGraph, partition::Partition)
    return induced_subgraph(graph, vertices(graph, partition))
  end

  function vertices(graph::PartitionedGraph, partition::Partition)
    return partitions(graph)[partition]
  end

  function blockdiag(graph1::PartitionedGraph, graph2::PartitionedGraph)
    new_parent_graph = blockdiag(parent_graph(graph1), parent_graph(graph2))
    new_partitions = copy(partitions(graph1))
    for k in keys(partitions(graph2))
      if haskey(new_partitions, k)
        append!(new_partitions[k], partitions(graph2)[k])
      else
        insert!(new_partitions, k, partitions(graph2)[k])
      end
    end
    return PartitionedGraph(new_parent_graph, new_partitions)
  end

  for f in [
    :edges,
    :vertices,
    :nv,
    :ne,
    :adjacency_matrix,
    :inneighbors,
    :outneighbors,
    :all_neighbors,
    :neighbors,
    :has_edge,
    :has_vertex
  ]
    @eval begin
      $f(graph::PartitionedGraph, args...) = $f(parent_graph(graph), args...)
    end
  end

  # Ambiguity errors with Graphs.jl
  for f in [
    :neighbors, :inneighbors, :outneighbors, :all_neighbors
  ]
    @eval begin
      $f(tn::PartitionedGraph, vertex::Integer) = $f(parent_graph(tn), vertex)
    end
  end

  is_directed(PG::Type{<:PartitionedGraph}) = is_directed(parent_graph_type(PG))

  function show(io::IO, mime::MIME"text/plain", graph::PartitionedGraph)
    println(io, "PartitionedGraph with partitions:")
    for k in keys(partitions(graph))
      println(io, "\"", k, "\" => ", partitions(graph)[k])
    end
    println(io)
    println(io, "and edges:")
    for e in edges(graph)
      show(io, mime, e)
      println(io)
    end
    return nothing
  end
end
