function construct_initial_mts(tn::ITensorNetwork, nvertices_per_subgraph::Integer; subgraph_kwargs=(;), kwargs...)
  return construct_initial_mts(tn, subgraphs(tn, nvertices_per_subgraph; subgraph_kwargs...); kwargs...)
end

function construct_initial_mts(tn::ITensorNetwork, subgraphs::DataGraph; init)
  # TODO: This is dropping the vertex data for some reason.
  # mts = DataGraph{vertextype(subgraphs),vertex_data_type(subgraphs),ITensor}(subgraphs)
  mts = DataGraph{vertextype(subgraphs),vertex_data_type(subgraphs),ITensor}(directed_graph(underlying_graph(subgraphs)), vertex_data(subgraphs))
  for subgraph in vertices(subgraphs)
    tns_to_contract = ITensor[]
    for subgraph_neighbor in neighbors(subgraphs, subgraph)
      edge_inds = Index[]
      for vertex in subgraphs[subgraph]
        psiv = tn[vertex]
        for e in [edgetype(tn)(vertex => neighbor) for neighbor in neighbors(tn, vertex)]
            if (find_subgraph(dst(e), subgraphs) == subgraph_neighbor)
            append!(edge_inds, commoninds(tn, e))
          end
        end
      end
      mt = itensor([init(Tuple(I)...) for I in CartesianIndices(tuple(dim.(edge_inds)...))], edge_inds)
      normalize!(mt)
      mts[subgraph => subgraph_neighbor] = mt
    end
  end
  return mts
end

"""
DO a single update of a message tensor using the current subgraph and the incoming mts
"""
function update_mt(tn::ITensorNetwork, subgraph_vertices::Vector, mts::Vector{ITensor}; contraction_sequence::Function=tn -> contraction_sequence(tn; alg="optimal")
)
  contract_list = [mts; [tn[v] for v in subgraph_vertices]]
  M = contract(contract_list; sequence=contraction_sequence(contract_list))
  normalize!(M)
  return M
end

"""
Do an update of all message tensors for a given flat ITensornetwork and its partition into sub graphs
"""
function update_all_mts(
  tn::ITensorNetwork,
  mts::DataGraph;
  contraction_sequence::Function=tn -> contraction_sequence(tn; alg="optimal")
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
  contraction_sequence::Function=tn -> contraction_sequence(tn; alg="optimal")
)
  for i in 1:niters
    mts = update_all_mts(tn, mts; contraction_sequence)
  end
  return mts
end

"""
given two flat networks psi and psi0, calculate the ratio of their contraction centred on the the subgraph containing v. The message tensors should be formulated over psi
Link indices between psi and psi0 should be consistent so the mts can be applied to both
"""
function get_single_site_expec(
  tn::ITensorNetwork,
  subgraphs::DataGraph,
  tnO::ITensorNetwork,
  v;
  contraction_sequence::Function=tn -> ITensorNetworks.contraction_sequence(tn; alg="optimal")
)
  subgraph = find_subgraph(v, subgraphs)
  connected_subgraphs = neighbors(subgraphs, subgraph)
  num_tensors_to_contract = ITensor[]
  denom_tensors_to_contract = ITensor[]
  for k in connected_subgraphs
    push!(num_tensors_to_contract, subgraphs[k => subgraph])
    push!(denom_tensors_to_contract, subgraphs[k => subgraph])
  end
  for vertex in subgraphs[subgraph]
    tnv = tn[vertex]
    tnvO = tnO[vertex]
    push!(num_tensors_to_contract, tnvO)
    push!(denom_tensors_to_contract, tnv)
  end
  numerator = ITensors.contract(num_tensors_to_contract; sequence=contraction_sequence(num_tensors_to_contract))[]
  denominator = ITensors.contract(denom_tensors_to_contract; sequence=contraction_sequence(denom_tensors_to_contract))[]
  return numerator / denominator
end

"""
given two flat networks psi and psi0, calculate the ratio of their contraction centred on the subgraph(s) containing v1 and v2. The message tensors should be formulated over psi.
"""
function get_two_site_expec(
  psi::ITensorNetwork,
  subgraphs::DataGraph,
  psiO::ITensorNetwork,
  v1,
  v2;
  contraction_sequence::Function=tn -> contraction_sequence(tn; alg="optimal")
)
  subgraph1 = find_subgraph(v1, subgraphs)
  subgraph2 = find_subgraph(v2, subgraphs)
  num_tensors_to_contract = ITensor[]
  denom_tensors_to_contract = ITensor[]

  if (subgraph1 == subgraph2)
    connected_subgraphs = neighbors(subgraphs, subgraph1)
    for k in connected_subgraphs
      push!(num_tensors_to_contract, subgraphs[k => subgraph1])
      push!(denom_tensors_to_contract, subgraphs[k => subgraph1])
    end

    for vertex in subgraphs[subgraph1]
      if (vertex != v1 && vertex != v2)
        push!(num_tensors_to_contract, deepcopy(psi[vertex]))
      else
        push!(num_tensors_to_contract, deepcopy(psiO[vertex]))
      end
      push!(denom_tensors_to_contract, deepcopy(psi[vertex]))
    end
  else
    connected_subgraphs1 = neighbors(subgraphs, subgraph1)
    for k in connected_subgraphs1
      if (k != subgraph2)
        push!(num_tensors_to_contract, subgraphs[k => subgraph1])
        push!(denom_tensors_to_contract, subgraphs[k => subgraph1])
      end
    end

    connected_subgraphs2 = neighbors(subgraphs, subgraph2)
    for k in connected_subgraphs2
      if (k != subgraph1)
        push!(num_tensors_to_contract, subgraphs[k => subgraph2])
        push!(denom_tensors_to_contract, subgraphs[k => subgraph2])
      end
    end

    for vertex in subgraphs[subgraph1]
      if (vertex != v1)
        push!(num_tensors_to_contract, deepcopy(psi[vertex]))
      else
        push!(num_tensors_to_contract, deepcopy(psiO[vertex]))
      end

      push!(denom_tensors_to_contract, deepcopy(psi[vertex]))
    end

    for vertex in subgraphs[subgraph2]
      if (vertex != v2)
        push!(num_tensors_to_contract, deepcopy(psi[vertex]))
      else
        push!(num_tensors_to_contract, deepcopy(psiO[vertex]))
      end

      push!(denom_tensors_to_contract, deepcopy(psi[vertex]))
    end
  end

  numerator = ITensors.contract(num_tensors_to_contract; sequence=contraction_sequence(num_tensors_to_contract))[1]
  denominator = ITensors.contract(denom_tensors_to_contract; sequence=contraction_sequence(denom_tensors_to_contract))[1]
  out = numerator / denominator

  return out
end

"""
Starting with initial guess for messagetensors, monitor the convergence of an observable on a single site v (which is emedded in tnO)
"""
function iterate_single_site_expec(
  tn::ITensorNetwork,
  subgraphs::DataGraph,
  niters::Int,
  tnO::ITensorNetwork,
  v,
)
  println(
    "Initial Guess for Observable on site " *
    string(v) *
    " is " *
    string(get_single_site_expec(tn, subgraphs, tnO, v)),
  )
  for i in 1:niters
    subgraphs = update_all_mts(tn, subgraphs, niters)
    approx_O = get_single_site_expec(tn, subgraphs, tnO, v)
    println(
      "After iteration " *
      string(i) *
      " Belief propagation gives observable on site " *
      string(v) *
      " is " *
      string(approx_O),
    )
  end
  return subgraphs
end

"""
Starting with initial guess for messagetensors, monitor the convergence of an observable on a pair of sites v1 and v2 (which is emedded in tnO)
"""
function iterate_two_site_expec(
  tn::ITensorNetwork,
  subgraphs::DataGraph,
  niters::Int,
  tnO::ITensorNetwork,
  v1,
  v2,
)
  println(
    "Initial Guess for Observable on sites " *
    string(v1) * " and " *string(v2) *
    " is " *
    string(get_two_site_expec(tn, subgraphs, tnO, v1, v2)),
  )
  for i in 1:niters
    subgraphs = update_all_mts(tn, subgraphs, niters)
    approx_O = get_two_site_expec(tn, subgraphs, tnO, v1, v2)
    println(
      "After iteration " *
      string(i) *
      " Belief propagation gives observable on site " *
      string(v1) * " and " * string(v2) *
      " is " *
      string(approx_O),
    )
  end
  return subgraphs
end

"""
Get two_site_rdm using belief propagation messagetensors
"""
function two_site_rdm_bp(
  tn::ITensorNetwork,
  psi::ITensorNetwork,
  subgraphs::DataGraph,
  v1,
  v2,
  s::IndsNetwork,
  combiners::DataGraph,
)
  subgraph1 = find_subgraph(v1, subgraphs)
  subgraph2 = find_subgraph(v2, subgraphs)
  tensors_to_contract = ITensor[]

  connected_subgraphs = neighbors(subgraphs, subgraph1)
  for k in connected_subgraphs
    if(k != subgraph2)
      push!(tensors_to_contract, subgraphs[k => subgraph1])
    end
  end

  for vertex in subgraphs[subgraph1]
    if (vertex != v1 && vertex != v2)
      push!(tensors_to_contract, deepcopy(tn[vertex]))
    end
  end

  if(subgraph2 != subgraph1)
    connected_subgraphs2 = neighbors(subgraphs, subgraph2)
    for k in connected_subgraphs2
      if (k != subgraph1)
        push!(tensors_to_contract, subgraphs[k => subgraph2])
      end
    end

    for vertex in subgraphs[subgraph2]
      if (vertex != v2)
        push!(tensors_to_contract, deepcopy(tn[vertex]))
      end
    end
  end

  psi1 = deepcopy(psi[v1])*prime!(dag(deepcopy(psi[v1])))
  for v in neighbors(psi, v1)
    C = haskey(combiners,NamedEdge(v => v1)) ? combiners[NamedEdge(v => v1)] : combiners[NamedEdge(v1 => v)]
    psi1 = psi1*C
  end
  push!(tensors_to_contract, psi1)

  psi2 = deepcopy(psi[v2])*prime!(dag(deepcopy(psi[v2])))
  for v in neighbors(psi, v2)
    C = haskey(combiners,NamedEdge(v => v2)) ? combiners[NamedEdge(v => v2)] : combiners[NamedEdge(v2 => v)]
    psi2 = psi2*C
  end
  push!(tensors_to_contract, psi2)

  rdm = ITensors.contract(tensors_to_contract)
  C = combiner(s[v1], s[v2])
  Cp = combiner(s[v1]', s[v2]')
  rdm = rdm * C * Cp
  tr_rdm = rdm * delta(combinedind(C), combinedind(Cp))
  return array(rdm / tr_rdm)
end
