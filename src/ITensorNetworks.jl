module ITensorNetworks

  # Special struct indicating a constructor is internal
  struct InternalConstructor end

  # General functions
  _not_implemented() = error("Not implemented")

  import Base: convert, copy

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
  using .DataGraphs: assign_data

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
  export IndsNetwork, ITensorNetwork, itensors

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

  # Helper functions
  function vertex_tag(v::Tuple)
    return "$(v[1])×$(v[2])"
  end

  function edge_tag(e)
    return "$(vertex_tag(src(e)))↔$(vertex_tag(dst(e)))"
  end

  function vertex_index(v, vertex_space)
    return Index(vertex_space; tags=vertex_tag(v))
  end

  function edge_index(e, edge_space)
    return Index(edge_space; tags=edge_tag(e))
  end

  #
  # AbstractIndsNetwork
  #

  abstract type AbstractIndsNetwork{I,V} <: AbstractDataGraph{Vector{I},Vector{I},V,CustomVertexEdge{V}} end

  # Field access
  # TODO: Only define for concrete type `IndsNetwork`.
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

  const UniformDataGraph{D,V} = DataGraph{D,D,V,CustomVertexEdge{V},CustomVertexGraph{V,Graphs.Graph{Int},Bijection{V,Int}}}

  struct IndsNetwork{I,V} <: AbstractIndsNetwork{I,V}
    parent_graph::UniformDataGraph{Vector{I},V}
  end

  function IndsNetwork(g::CustomVertexGraph, link_space::Nothing, site_space::Nothing)
    dg = DataGraph{Vector{Index},Vector{Index}}(g)
    return IndsNetwork(dg)
  end

  function IndsNetwork(g::CustomVertexGraph, link_space, site_space)
    is = IndsNetwork(g, nothing, nothing)
    for e in edges(is)
      is[e] = [Index(link_space, edge_tag(e))]
    end
    for v in vertices(is)
      is[v] = [Index(site_space, vertex_tag(v))]
    end
    return is
  end

  function IndsNetwork(g::CustomVertexGraph; link_space=nothing, site_space=nothing)
    return IndsNetwork(g, link_space, site_space)
  end

  copy(is::IndsNetwork) = IndsNetwork(copy(parent_graph(is)))

  #
  # AbstractITensorNetwork
  #

  abstract type AbstractITensorNetwork{V} <: AbstractDataGraph{ITensor,ITensor,V,CustomVertexEdge{V}} end

  # Field access
  # TODO: Only define for concrete type `ITensorNetwork`.
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
    parent_graph::UniformDataGraph{ITensor,V}
  end

  copy(is::ITensorNetwork) = ITensorNetwork(copy(parent_graph(is)))

  # TODO: Add sitespace, linkspace
  function ITensorNetwork(g::CustomVertexGraph)
    dg = DataGraph{ITensor,ITensor}(g)
    return ITensorNetwork(dg)
  end

  # Assigns indices with space `link_space` to unassigned
  # edges of the network.
  function _ITensorNetwork(is::IndsNetwork, link_space)
    edge_data(e) = [edge_index(e, link_space)]
    is_assigned = assign_data(is; edge_data)
    return _ITensorNetwork(is_assigned, nothing)
  end

  function _ITensorNetwork(is::IndsNetwork, link_space::Nothing)
    g = parent_graph(parent_graph(is))
    tn = ITensorNetwork(g)
    for v in vertices(tn)
      siteinds = is[v]
      linkinds = [is[v => nv] for nv in neighbors(is, v)]
      tn[v] = ITensor(siteinds, linkinds...)
    end
    return tn
  end

  function ITensorNetwork(is::IndsNetwork; link_space=nothing)
    return _ITensorNetwork(is, link_space)
  end

  # Convert to a collection of ITensors (`Vector{ITensor}`).
  function itensors(tn::ITensorNetwork)
    return collect(vertex_data(tn))
  end

end
