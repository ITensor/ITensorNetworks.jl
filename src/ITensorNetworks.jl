module ITensorNetworks

  using Dictionaries
  using ITensors
  using ITensors.ITensorVisualizationCore
  using Graphs

  include(joinpath("SubIndexing", "src", "SubIndexing.jl"))
  using .SubIndexing

  include("CustomVertexGraphs/src/CustomVertexGraphs.jl")
  using .CustomVertexGraphs

  using .CustomVertexGraphs: Bijection, CustomVertexEdge, CustomVertexGraph

  include("DataGraphs/src/DataGraphs.jl")
  using .DataGraphs
  using .DataGraphs: assign_data

  include("imports.jl")

  # General functions
  _not_implemented() = error("Not implemented")

  # When setting an edge with collections of `Index`, set the reverse direction
  # edge with the `dag`.
  DataGraphs.reverse_direction(is::Union{Index,Tuple{Vararg{<:Index}},Vector{<:Index}}) = dag(is)

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

  function CustomVertexGraph(itensors::Vector{ITensor})
    return set_vertices(Graph(itensors), 1:length(itensors))
  end

  front(itr, n = 1) = Iterators.take(itr, length(itr) - n)
  tail(itr) = Iterators.drop(itr, 1)

  # Helper functions
  vertex_tag(v::Int) = "$v"

  function vertex_tag(v::Tuple)
    t = "$(first(v))"
    for vn in Base.tail(v)
      t *= "×$vn"
    end
    return t
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

  const UniformDataGraph{D,V} = DataGraph{D,D,V,CustomVertexEdge{V},CustomVertexGraph{V,Graphs.Graph{Int},Bijection{V,Int}}}

  include("lattices.jl")
  include("abstractindsnetwork.jl")
  include("indsnetwork.jl")
  include("sitetype.jl")
  include("abstractitensornetwork.jl")
  include("itensornetwork.jl")

  include("exports.jl")
end
