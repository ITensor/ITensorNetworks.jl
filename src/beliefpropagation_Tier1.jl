function construct_initial_mts(
  tn::ITensorNetwork, nvertices_per_partition::Integer; partition_kwargs=(;), kwargs...
)
  return construct_initial_mts(
    tn, partition(tn; nvertices_per_partition, partition_kwargs...); kwargs...
  )
end

function construct_initial_mts(tn::ITensorNetwork, subgraphs::DataGraph; init)
  # TODO: This is dropping the vertex data for some reason.
  # mts = DataGraph{vertextype(subgraphs),vertex_data_type(subgraphs),ITensor}(subgraphs)
  mts = DataGraph{vertextype(subgraphs),vertex_data_type(subgraphs),ITensor}(
    directed_graph(underlying_graph(subgraphs))
  )
  for v in vertices(mts)
    mts[v] = subgraphs[v]
  end
  for subgraph in vertices(subgraphs)
    tns_to_contract = ITensor[]
    for subgraph_neighbor in neighbors(subgraphs, subgraph)
      edge_inds = Index[]
      for vertex in vertices(subgraphs[subgraph])
        psiv = tn[vertex]
        for e in [edgetype(tn)(vertex => neighbor) for neighbor in neighbors(tn, vertex)]
          if (find_subgraph(dst(e), subgraphs) == subgraph_neighbor)
            append!(edge_inds, commoninds(tn, e))
          end
        end
      end
      mt = normalize!(
        itensor(
          [init(Tuple(I)...) for I in CartesianIndices(tuple(dim.(edge_inds)...))],
          edge_inds,
        ),
      )
      mts[subgraph => subgraph_neighbor] = mt
    end
  end
  return mts
end

"""
DO a single update of a message tensor using the current subgraph and the incoming mts
"""
function update_mt(
  tn::ITensorNetwork,
  subgraph_vertices::Vector,
  mts::Vector{ITensor};
  contraction_sequence::Function=tn -> contraction_sequence(tn; alg="optimal"),
)
  contract_list = [mts; [tn[v] for v in subgraph_vertices]]
  new_mt = if isone(length(contract_list))
    copy(only(contract_list))
  else
    contract(contract_list; sequence=contraction_sequence(contract_list))
  end
  return normalize!(new_mt)
end

function update_mt(
  tn::ITensorNetwork, subgraph::ITensorNetwork, mts::Vector{ITensor}; kwargs...
)
  return update_mt(tn, vertices(subgraph), mts; kwargs...)
end

"""
Do an update of all message tensors for a given ITensornetwork and its partition into sub graphs
"""
function update_all_mts(
  tn::ITensorNetwork,
  mts::DataGraph;
  contraction_sequence::Function=tn -> contraction_sequence(tn; alg="optimal"),
)
  update_mts = copy(mts)
  for e in edges(mts)
    environment_tensors = ITensor[]
    for neighboring_subgraph in neighbors(mts, src(e))
      if (neighboring_subgraph != dst(e))
        push!(environment_tensors, mts[neighboring_subgraph => src(e)])
      end
    end
    update_mts[src(e) => dst(e)] = update_mt(
      tn, mts[src(e)], environment_tensors; contraction_sequence
    )
  end
  return update_mts
end

function update_all_mts(
  tn::ITensorNetwork,
  mts::DataGraph,
  niters::Int;
  contraction_sequence::Function=tn -> contraction_sequence(tn; alg="optimal"),
)
  for i in 1:niters
    mts = update_all_mts(tn, mts; contraction_sequence)
  end
  return mts
end

#Given a subet of vertices of a given Tensor Network and the Message Tensors for that network, return a Dictionary with the involved subgraphs as keys and the vector of tensors associated with that subgraph as values
#Specifically, the contraction of the environment tensors and tn[vertices] will be a scalar.
function get_environment_BP(tn::ITensorNetwork, mts::DataGraph, verts::Vector)
  environment_tensors = Dict{Any, Vector{ITensor}}()
  subgraphs = unique([find_subgraph(v, mts) for v in verts])
  for s in subgraphs
    subgraph_tensors = ITensor[]
    for neighboring_subgraph in neighbors(mts, s)
      if(neighboring_subgraph ∉ subgraphs)
        push!(subgraph_tensors, mts[neighboring_subgraph => s])
      end
    end

    for v in vertices(mts[s])
      if(v ∉ verts)
        push!(subgraph_tensors, tn[v])
      end
    end

    environment_tensors[s] = subgraph_tensors
  end

  return environment_tensors

end

#Version of the above without the message tensors already preformed. Can either specify a vertex_grouping or just a nvertices_per_partition
function get_environment_BP(tn::ITensorNetwork, verts::Vector; nvertices_per_partition=1, niters=10, vertex_groups = nothing)
  mts = compute_message_tensors(tn; nvertices_per_partition=nvertices_per_partition, niters=niters, vertex_groups = vertex_groups)
  return get_environment_BP(tn, mts, verts)
  
end

#Function to calculate the contraction of a tensor network centred on the vertices verts. Using message tensors.
function calculate_contraction_BP(tn::ITensorNetwork, mts::DataGraph, verts::Vector, verts_tensors::Vector{ITensor})
  environment_tensors = get_environment_BP(tn, mts, verts)
  environment_tensors = flatten([environment_tensors[i] for i in keys(environment_tensors)])
  return contract(vcat(environment_tensors, verts_tensors); sequence = contraction_sequence(vcat(environment_tensors, verts_tensors)))
end

#Function to initialise and update message tensors of a tensornetwork
function compute_message_tensors(tn::ITensorNetwork; niters = 10, nvertices_per_partition = 1, vertex_groups = nothing)
  if(vertex_groups == nothing)
    Z = partition(tn; nvertices_per_partition = nvertices_per_partition)
  else
    Z = partition(tn, vertex_groups)
  end

  mts = construct_initial_mts(tn, Z; init=(I...) -> @compat allequal(I) ? 1 : 0)
  mts = update_all_mts(tn, mts, niters)
  return mts
end