module ITensorNetworks

  # General functions
  _not_implemented() = error("Not implemented")

  using Dictionaries
  using ITensors
  using Graphs

  include(joinpath("SubIndexing", "src", "SubIndexing.jl"))
  using .SubIndexing

  include("CustomVertexGraphs/src/CustomVertexGraphs.jl")
  using .CustomVertexGraphs

  using .CustomVertexGraphs: Bijection, CustomVertexEdge, CustomVertexGraph

  include("DataGraphs/src/DataGraphs.jl")
  using .DataGraphs
  import .DataGraphs: parent_graph, vertex_data, edge_data

  # When setting an edge with collections of `Index`, set the reverse direction
  # edge with the `dag`.
  DataGraphs.reverse_direction(is::Union{Index,Tuple{Vararg{<:Index}},Vector{<:Index}}) = dag(is)

  # Graphs
  import Graphs: Graph
  export grid, edges, vertices, ne, nv, src, dst, neighbors, has_edge, has_vertex

  # CustomVertexGraphs
  export set_vertices

  # DataGraphs
  export DataGraph

  # ITensorNetworks
  export IndsNetwork, ITensorNetwork

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

  #
  # AbstractIndsNetwork
  #

  abstract type AbstractIndsNetwork{I,V} <: AbstractDataGraph{Vector{I},Vector{I},V,CustomVertexEdge{V}} end

  # Field access
  parent_graph(graph::AbstractIndsNetwork) = getfield(graph, :parent_graph)

  # AbstractDataGraphs overloads
  for f in [:vertex_data, :edge_data]
    @eval begin
      $f(graph::AbstractIndsNetwork, args...) = $f(parent_graph(graph), args...)
    end
  end

  #
  # IndsNetwork
  #

  struct IndsNetwork{I,V} <: AbstractIndsNetwork{I,V}
    parent_graph::DataGraph{Vector{I},Vector{I},V,CustomVertexEdge{V},CustomVertexGraph{V,Graphs.Graph{Int},Bijection{V,Int}}}
  end

  function IndsNetwork(g::AbstractGraph; linkspace=nothing, sitespace=nothing)
    g = DataGraph{Vector{Index},Vector{Index}}(g)
    return IndsNetwork(g)
  end

  #
  # AbstractITensorNetwork
  #

  abstract type AbstractITensorNetwork{V} <: AbstractDataGraph{ITensor,ITensor,V,CustomVertexEdge{V}} end

  # Field access
  parent_graph(graph::AbstractITensorNetwork) = getfield(graph, :parent_graph)

  # AbstractDataGraphs overloads
  for f in [:vertex_data, :edge_data]
    @eval begin
      $f(graph::AbstractITensorNetwork, args...) = $f(parent_graph(graph), args...)
    end
  end

  #
  # ITensorNetwork
  #

  struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
    parent_graph::DataGraph{ITensor,ITensor,V,CustomVertexEdge{V},CustomVertexGraph{V,Graph{Int},Bijection{V,Int}}}
  end

  ## function IndsNetwork(g::AbstractGraph; linkspace=nothing, sitespace=nothing)
  ##   g = DataGraph{Vector{Index},Vector{Index}}(g)
  ##   return IndsNetwork(g)
  ## end

end
