module ITensorNetworks

  # General functions
  _not_implemented() = error("Not implemented")

  using Dictionaries
  using ITensors
  using Graphs

  include("CustomVertexGraphs/src/CustomVertexGraphs.jl")
  using .CustomVertexGraphs

  include("DataGraphs/src/DataGraphs.jl")
  using .DataGraphs

  # Graphs
  export grid, edges, vertices, ne, nv, src, dst, neighbors

  # CustomVertexGraphs
  export set_vertices

  # DataGraphs
  export DataGraph

  # ITensorNetworks
  export Network

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

  #const Network{D,V} = DataGraph{D,D,V,CustomVertexEdge{V},CustomVertexGraph{V,Graph{Int},Bijection{V,Int}}}

  # IndsNetwork
  #const IndsNetwork{V,Index{T}} = DataVertexGraph{V,Index{T}}

  # ITensorNetwork
  #const ITensorNetwork{V} = DataVertexGraph{V,ITensor}

end
