module ITensorNetworks

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
  import .DataGraphs: underlying_graph, vertex_data, edge_data
  using .DataGraphs: assign_data

  # When setting an edge with collections of `Index`, set the reverse direction
  # edge with the `dag`.
  DataGraphs.reverse_direction(is::Union{Index,Tuple{Vararg{<:Index}},Vector{<:Index}}) = dag(is)

  # Graphs
  import Graphs: Graph
  export grid, edges, vertices, ne, nv, src, dst, neighbors, has_edge, has_vertex

  # ITensors
  import ITensors: siteinds, linkinds, uniqueinds, commoninds, prime

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
  data_graph(graph::AbstractIndsNetwork) = _not_implemented()

  # AbstractDataGraphs overloads
  for f in [:vertex_data, :edge_data]
    @eval begin
      $f(graph::AbstractIndsNetwork, args...) = $f(data_graph(graph), args...)
    end
  end

  #
  # IndsNetwork
  #

  const UniformDataGraph{D,V} = DataGraph{D,D,V,CustomVertexEdge{V},CustomVertexGraph{V,Graphs.Graph{Int},Bijection{V,Int}}}

  # TODO: Rename field to `data_graph`.
  struct IndsNetwork{I,V} <: AbstractIndsNetwork{I,V}
    data_graph::UniformDataGraph{Vector{I},V}
  end
  data_graph(is::IndsNetwork) = getfield(is, :data_graph)
  underlying_graph(is::IndsNetwork) = underlying_graph(data_graph(is))

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

  copy(is::IndsNetwork) = IndsNetwork(copy(data_graph(is)))

  #
  # AbstractITensorNetwork
  #

  abstract type AbstractITensorNetwork{V} <: AbstractDataGraph{ITensor,ITensor,V,CustomVertexEdge{V}} end

  # Field access
  data_graph(graph::AbstractITensorNetwork) = _not_implemented()

  # AbstractDataGraphs overloads
  for f in [:vertex_data, :edge_data]
    @eval begin
      $f(graph::AbstractITensorNetwork, args...) = $f(data_graph(graph), args...)
    end
  end

  #
  # ITensorNetwork
  #

  # TODO: Rename field to `data_graph`.
  struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
    data_graph::UniformDataGraph{ITensor,V}
  end
  data_graph(tn::ITensorNetwork) = getfield(tn, :data_graph)
  underlying_graph(tn::ITensorNetwork) = underlying_graph(data_graph(tn))

  copy(tn::ITensorNetwork) = ITensorNetwork(copy(data_graph(tn)))

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
    g = underlying_graph(is)
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

  function incident_edges(g::AbstractGraph{V}, v::V) where {V}
    return [edgetype(g)(v, vn) for vn in neighbors(g, v)]
  end

  # Convert to an IndsNetwork
  function IndsNetwork(tn::ITensorNetwork)
    is = IndsNetwork(underlying_graph(tn))
    for v in vertices(tn)
      is[v] = siteinds(tn, v)
      for e in incident_edges(tn, v)
        append!(is[v], linkinds(tn, e))
      end
    end
    return is
  end

  #
  # Index access
  #

  function uniqueinds(tn::ITensorNetwork{V}, v::V) where {V}
    is = Index[]
    for vn in neighbors(tn, v)
      append!(is, uniqueinds(tn[v], tn[vn]))
    end
    return is
  end

  function siteinds(tn::ITensorNetwork{V}, v::V) where {V}
    return uniqueinds(tn, v)
  end

  function commoninds(tn::ITensorNetwork{V}, e::CustomVertexEdge{V}) where {V}
    return commoninds(tn[src(e)], tn[dst(e)])
  end

  function linkinds(tn::ITensorNetwork{V}, e::CustomVertexEdge{V}) where {V}
    return commoninds(tn, e)
  end

  # Priming and tagging (changing Index identifiers)
  function replaceinds(tn::ITensorNetwork, is_is′::Pair{<:IndsNetwork,<:IndsNetwork})
    tn = copy(tn)
    is, is′ = is_is′
    # TODO: Check that `is` and `is′` have the same vertices and edges.
    for v in vertices(is)
      setindex_preserve_graph!(tn, replaceinds(tn[v], is[v] => is′[v]), v)
    end
    for e in edges(is)
      setindex_preserve_graph!(tn, replaceinds(tn[v], is[e] => is′[e]), e)
    end
    return tn
  end

  function prime(tn::IndsNetwork)
    _not_implemented()
  end

  function prime(tn::ITensorNetwork)
    is = IndsNetwork(tn)
    is′ = prime(is)
    return replaceinds(tn, is, is′)
  end

end
