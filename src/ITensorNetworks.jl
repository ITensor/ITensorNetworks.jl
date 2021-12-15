module ITensorNetworks

  using Dictionaries
  using ITensors
  using Graphs

  import Graphs: Graph, vertices, neighbors, outneighbors, inneighbors, all_neighbors

  include("CustomVertexGraphs/src/CustomVertexGraphs.jl")
  using .CustomVertexGraphs
  import .CustomVertexGraphs: parent_vertex, parent_graph
  export CustomVertexGraph, CustomVertexDiGraph, CustomVertexEdge

  include("PartitionedGraphs/src/PartitionedGraphs.jl")
  using .PartitionedGraphs
  import .PartitionedGraphs: partitions, set_partitions
  export PartitionedGraph, PartitionedDiGraph, Partition, partitions, set_partitions, subgraph

  include("indsnetwork.jl")

  # Overloads of CustomVertexGraphs functions
  parent_graph(pg::PartitionedGraph) = pg.graph
  parent_vertex(pg::PartitionedGraph, vertex) = parent_vertex(parent_graph(pg), vertex)

  import Base: getindex, show
  import Graphs: Graph, edges, nv, ne, outneighbors, has_edge, add_edge!, induced_subgraph, src, dst, is_directed, adjacency_matrix

  export ITensorNetwork, tensor_product, ⊗

  function Graph(itensors::Vector{ITensor})
    nv_graph = length(itensors)
    graph = Graph(nv_graph)
    for i in 1:(nv_graph - 1), j in (i + 1):nv_graph
      if hascommoninds(itensors[i], itensors[j])
        add_edge!(graph, i => j)
      end
    end
    return graph
  end

  _not_implemented() = error("Not implemented")

  abstract type AbstractITensorNetwork{T} <: AbstractGraph{T} end

  graph(tn::AbstractITensorNetwork) = _not_implemented()
  itensors(tn::AbstractITensorNetwork) = _not_implemented()

  # General graph overloads
  for f in [
    :edges, :vertices, :nv, :ne, :neighbors, :inneighbors, :outneighbors, :all_neighbors, :has_edge, :adjacency_matrix, # AbstractGraphs
    :partitions, # PartitionedGraphs
  ]
    @eval begin
      $f(tn::AbstractITensorNetwork, args...) = $f(graph(tn), args...)
    end
  end

  # Ambiguity errors with Graphs.jl
  for f in [
    :neighbors, :inneighbors, :outneighbors, :all_neighbors
  ]
    @eval begin
      $f(tn::AbstractITensorNetwork, vertex::Integer) = $f(graph(tn), vertex)
    end
  end

  is_directed(::Type{<:AbstractITensorNetwork}) = false

  function set_partitions(tn::AbstractITensorNetwork, args...; kwargs...)
    new_graph = set_partitions(graph(tn), args...; kwargs...)
    return ITensorNetwork(itensors(tn), new_graph)
  end
  
  parent_vertex(tn::AbstractITensorNetwork, vertex) = parent_vertex(graph(tn), vertex)

  function get_itensor(tn::AbstractITensorNetwork, vertex)
    return itensors(tn)[parent_vertex(tn, vertex)]
  end

  function get_itensors(tn::AbstractITensorNetwork, vertices::AbstractVector)
    return [tn[vertex] for vertex in vertices]
  end

  function induced_subgraph(tn::AbstractITensorNetwork, partition::Partition)
    return induced_subgraph(tn, vertices(tn, partition))
  end

  function induced_subgraph(tn::AbstractITensorNetwork, vertices::AbstractVector)
    subgraph, vertices_map = induced_subgraph(graph(tn), vertices)
    new_itensors = get_itensors(tn, vertices)
    new_tn = ITensorNetwork(new_itensors, vertices)
    return new_tn, vertices_map
  end

  getindex(tn::AbstractITensorNetwork, vertex) = get_itensor(tn, vertex)

  Graph(tn::AbstractITensorNetwork) = parent_graph(parent_graph(graph(tn)))

  struct ITensorNetwork{T} <: AbstractITensorNetwork{T}
    itensors::Vector{ITensor}
    graph::PartitionedGraph{String,CustomVertexGraph{T,Graph{Int},Int},T}
  end

  graph(tn::ITensorNetwork) = tn.graph
  itensors(tn::ITensorNetwork) = tn.itensors

  function add_label(tn::ITensorNetwork, label)
    error("Not implemented")
  end

  # Combine two tensor networks into a single tensor network
  function tensor_product(tn1::ITensorNetwork, tn2::ITensorNetwork)
    new_itensors = vcat(itensors(tn1), itensors(tn2))
    new_graph = blockdiag(graph(tn1), graph(tn2))
    return ITensorNetwork(new_itensors, new_graph)
  end
  const ⊗ = tensor_product

  function default_partitions(g::AbstractGraph)
    d = Dictionary([""], [vertices(g)])
    return d
  end

  function ITensorNetwork(itensors::Vector{ITensor}, lg::CustomVertexGraph; partitions=default_partitions(lg))
    return ITensorNetwork(itensors, PartitionedGraph(lg, partitions))
  end

  function ITensorNetwork(itensors::Vector{ITensor}, graph::Graph, labels=Vector(1:nv(graph)); kwargs...)
    return ITensorNetwork(itensors, CustomVertexGraph(graph, labels); kwargs...)
  end

  function ITensorNetwork(nv::Integer; kwargs...)
    return ITensorNetwork([ITensor() for _ in 1:nv], Graph(nv); kwargs...)
  end

  function ITensorNetwork(nv::Tuple{Vararg{Integer}}; kwargs...)
    return ITensorNetwork([ITensor() for _ in 1:prod(nv)], CustomVertexGraph(nv); kwargs...)
  end

  function ITensorNetwork(itensors::Vector{ITensor}; kwargs...)
    return ITensorNetwork(itensors, Graph(itensors); kwargs...)
  end

  function ITensorNetwork(itensors::Vector{ITensor}, labels::Vector; kwargs...)
    return ITensorNetwork(itensors, Graph(itensors), labels; kwargs...)
  end

  edge_tags(e::Edge) = TagSet("l=$(src(e))→$(dst(e))")

  function ITensorNetwork(graph::Graph, labels=Vector(1:nv(graph)); kwargs...)

    # TODO: Allow control over this.
    dim = 1

    nv_graph = nv(graph)
    itensors = [ITensor() for _ in 1:nv_graph]
    for e in edges(graph)
      i = Index(dim; tags=edge_tags(e))
      x = ITensor(i)
      itensors[src(e)] *= x
      itensors[dst(e)] *= dag(x)
    end
    return ITensorNetwork(itensors, graph, labels; kwargs...)
  end
  ITensorNetwork(graph::Graph{T}, labels=Vector(1:nv(graph)); kwargs...) where {T} = ITensorNetwork{T}(graph, labels; kwargs...)

  function tensor_product(t1::ITensor, t2::ITensor)
    return ITensorNetwork([t1, t2])
  end

  # TODO: This removes labels and partitions.
  function tensor_product(tn::ITensorNetwork, t::ITensor)
    new_itensors = vcat(itensors(tn), t)
    return ITensorNetwork(new_itensors)
  end

  # TODO: This removes labels and partitions.
  function tensor_product(t::ITensor, tn::ITensorNetwork)
    new_itensors = vcat(t, itensors(tn))
    return ITensorNetwork(new_itensors)
  end

  function show(io::IO, mime::MIME"text/plain", tn::ITensorNetwork)
    println(io, "ITensorNetwork with $(nv(tn)) tensors")
    println(io, "Network partitions:")
    for k in keys(partitions(tn))
      println(io, k, " => ", partitions(tn)[k])
    end
    println(io)
    println(io, "and $(ne(tn)) edges:")
    for e in edges(tn)
      show(io, mime, e)
      println(io)
    end
    return nothing
  end
  show(io::IO, tn::ITensorNetwork) = show(io, MIME"text/plain"(), tn)
end
