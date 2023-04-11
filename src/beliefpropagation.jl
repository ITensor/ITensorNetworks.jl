function construct_initial_mts(
  tn::ITensorNetwork, nvertices_per_partition::Integer; partition_kwargs=(;), kwargs...
)
  return construct_initial_mts(
    tn, partition(tn; nvertices_per_partition, partition_kwargs...); kwargs...
  )
end

function construct_initial_mts(
  tn::ITensorNetwork,
  subgraphs::DataGraph;
  contract_kwargs=(;),
  init=(I...) -> allequal(I) ? 1 : 0,
)
  mts = DataGraph{vertextype(subgraphs),vertex_data_type(subgraphs),ITensorNetwork}(
    directed_graph(underlying_graph(subgraphs))
  )
  for v in vertices(mts)
    mts[v] = subgraphs[v]
  end
  for subgraph in vertices(subgraphs)
    for subgraph_neighbor in neighbors(subgraphs, subgraph)
      boundary_vertices_s = setdiff(
        vertices(subgraphs[subgraph]),
        unique(
          flatten([
            neighbors(subgraphs[subgraph_neighbor], vn) for
            vn in vertices(subgraphs[subgraph_neighbor])
          ]),
        ),
      )
      boundary_vertices_sn = setdiff(
        vertices(subgraphs[subgraph_neighbor]),
        unique(
          flatten([
            neighbors(subgraphs[subgraph], vn) for vn in vertices(subgraphs[subgraph])
          ]),
        ),
      )
      boundary_tensors = ITensor[]
      for v in boundary_vertices_s
        inds_boundary = flatten(unique([inds(tn[vn]) for vn in boundary_vertices_sn]))
        inds_internal = flatten(
          unique([inds(tn[v]) for v in setdiff(boundary_vertices_s, [v])])
        )
        new_inds = intersect(inds(tn[v]), vcat(inds_boundary, inds_internal))
        push!(
          boundary_tensors,
          itensor(
            [init(Tuple(I)...) for I in CartesianIndices(tuple(dim.(new_inds)...))],
            new_inds,
          ),
        )
      end

      mt = ITensorNetwork(Dictionaries.Dictionary(boundary_vertices_s, boundary_tensors))

      contract_output = contract(mt; contract_kwargs...)
      mts[subgraph => subgraph_neighbor] = if typeof(contract_output) == ITensor
        ITensorNetwork(contract_output)
      else
        first(contract_output)
      end
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
  mts::Vector{ITensorNetwork};
  contract_kwargs=(;),
)
  contract_list = ITensorNetwork[mts; ITensorNetwork([tn[v] for v in subgraph_vertices])]

  tn = if isone(length(contract_list))
    copy(only(contract_list))
  else
    reduce(⊗, contract_list)
  end

  contract_output = contract(tn; contract_kwargs...)
  itn = if typeof(contract_output) == ITensor
    ITensorNetwork(contract_output)
  else
    first(contract_output)
  end
  normalize!.(vertex_data(itn))

  return itn
end

function update_mt(
  tn::ITensorNetwork, subgraph::ITensorNetwork, mts::Vector{ITensorNetwork}; kwargs...
)
  return update_mt(tn, vertices(subgraph), mts; kwargs...)
end

"""
Do an update of all message tensors for a given ITensornetwork and its partition into sub graphs
"""
function update_all_mts(tn::ITensorNetwork, mts::DataGraph; contract_kwargs=(;))
  update_mts = copy(mts)
  for e in edges(mts)
    environment_tensornetworks = ITensorNetwork[
      mts[e_in] for e_in in setdiff(boundary_edges(mts, src(e); dir=:in), [reverse(e)])
    ]

    update_mts[src(e) => dst(e)] = update_mt(
      tn, mts[src(e)], environment_tensornetworks; contract_kwargs
    )
  end
  return update_mts
end

function update_all_mts(
  tn::ITensorNetwork, mts::DataGraph, niters::Int; contract_kwargs=(;)
)
  for i in 1:niters
    mts = update_all_mts(tn, mts; contract_kwargs)
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

  env_tns = ITensorNetwork[mts[e] for e in boundary_edges(mts, subgraphs; dir=:in)]
  central_tn = ITensorNetwork([
    tn[v] for v in setdiff(flatten([vertices(mts[s]) for s in subgraphs]), verts)
  ])
  return vcat(env_tns, ITensorNetwork[central_tn])
end

function get_environment(
  output_type::Type, tn::ITensorNetwork, mts::DataGraph, verts::Vector; kwargs...
)
  itns = get_environment(tn::ITensorNetwork, mts::DataGraph, verts::Vector; kwargs...)

  if output_type == Vector{ITensorNetwork}
    return itns
  else
    itn = reduce(⊗, itns)
    if output_type == ITensorNetwork
      return itn
    elseif output_type == Vector{ITensor}
      return ITensor[itn[v] for v in vertices(itn)]
    else
      error("Output Type for get_environment not Supported!")
    end
  end
end

"""
Calculate the contraction of a tensor network centred on the vertices verts. Using message tensors.
Defaults to using tn[verts] as the local network but can be overriden
"""
function calculate_contraction_network(
  tn::ITensorNetwork,
  mts::DataGraph,
  verts::Vector;
  verts_tn=ITensorNetwork([tn[v] for v in verts]),
)
  environment_tns = get_environment(tn, mts, verts)

  return reduce(⊗, vcat(environment_tns, ITensorNetwork[verts_tn]))
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
  contract_kwargs=(;),
  init_contract_kwargs=(;),
  init_kwargs...,
)
  Z = partition(tn; nvertices_per_partition, npartitions, subgraph_vertices=vertex_groups)

  mts = construct_initial_mts(tn, Z; contract_kwargs=init_contract_kwargs, init_kwargs...)
  mts = update_all_mts(tn, mts, niters; contract_kwargs)
  return mts
end
