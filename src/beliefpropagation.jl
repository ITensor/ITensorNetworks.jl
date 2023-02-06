function construct_initial_mts(
  tn::ITensorNetwork, nvertices_per_partition::Integer; partition_kwargs=(;), kwargs...
)
  return construct_initial_mts(
    tn, partition(tn; nvertices_per_partition, partition_kwargs...); kwargs...
  )
end

function construct_initial_mts(
  tn::ITensorNetwork, subgraphs::DataGraph; init=(I...) -> @compat allequal(I) ? 1 : 0
)
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
    environment_tensors = ITensor[
      mts[e_in] for e_in in setdiff(boundary_edges(mts, src(e); dir=:in), [reverse(e)])
    ]
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

"""
Given a subet of vertices of a given Tensor Network and the Message Tensors for that network, return a Dictionary with the involved subgraphs as keys and the vector of tensors associated with that subgraph as values
Specifically, the contraction of the environment tensors and tn[vertices] will be a scalar.
"""
function get_environment(tn::ITensorNetwork, mts::DataGraph, verts::Vector; dir=:in)
  subgraphs = unique([find_subgraph(v, mts) for v in verts])

  if dir == :out
    return get_environment(tn, mts, setdiff(vertices(tn), verts))
  end

  env_tensors = ITensor[mts[e] for e in boundary_edges(mts, subgraphs; dir=:in)]
  return vcat(
    env_tensors,
    ITensor[tn[v] for v in setdiff(flatten([vertices(mts[s]) for s in subgraphs]), verts)],
  )
end

"""
Calculate the contraction of a tensor network centred on the vertices verts. Using message tensors.
Defaults to using tn[verts] as the local network but can be overriden
"""
function calculate_contraction(
  tn::ITensorNetwork,
  mts::DataGraph,
  verts::Vector;
  verts_tensors=ITensor[tn[v] for v in verts],
  contraction_sequence::Function=tn -> contraction_sequence(tn; alg="optimal"),
)
  environment_tensors = get_environment(tn, mts, verts)
  tensors_to_contract = vcat(environment_tensors, verts_tensors)
  return contract(tensors_to_contract; sequence=contraction_sequence(tensors_to_contract))
end

"""
Simulaneously initialise and update message tensors of a tensornetwork
"""
function compute_message_tensors(
  tn::ITensorNetwork;
  niters=10,
  nvertices_per_partition=nothing,
  npartitions=nothing,
  vertex_groups=nothing,
  kwargs...,
)
  Z = partition(tn; nvertices_per_partition, npartitions, subgraph_vertices=vertex_groups)

  mts = construct_initial_mts(tn, Z; kwargs...)
  mts = update_all_mts(tn, mts, niters)
  return mts
end
