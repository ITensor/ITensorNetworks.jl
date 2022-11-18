#Construct the random initial Message Tensors for an ITensor Network, based on a partitioning into subgraphs specified ny 'sub graphs'
#The ITensorNetwork needs to be flat (i.e. just sites and link indices, no site indices), and is assumed to only have 1 link indice between any two sites
#init=(I...) -> allequal(I) ? 1 : 0 (identity), init=(I...) -> randn() (normally distributed random), etc...
function construct_initial_mts(flatpsi::ITensorNetwork, dg_subgraphs::DataGraph; init
)
  mts = Dict{Pair,ITensor}()

  for i in vertices(dg_subgraphs)
    tns_to_contract = ITensor[]
    for j in neighbors(dg_subgraphs, i)
      edge_inds = Index[]
      for vertex in dg_subgraphs[i]
        psiv = flatpsi[vertex]
        for e in NamedDimEdge.(Ref(vertex) .=> neighbors(flatpsi, vertex))
          if (find_subgraph(dst(e), dg_subgraphs) == j)
            edge_ind = commoninds(flatpsi, e)[1]
            push!(edge_inds, edge_ind)
          end
        end
      end
      X1 = itensor([init(Tuple(I)...) for I in CartesianIndices(tuple(dim.(edge_inds)...))], edge_inds)
      normalize!(X1)
      mts[i => j] = X1
    end
  end

  return mts
end

#DO a single update of a message tensor using the current subgraph and the incoming mts
function updatemt(flatpsi::ITensorNetwork, subgraph::Vector{Tuple}, mts::Vector{ITensor}; contraction_sequence::Function=tn -> ITensorNetworks.contraction_sequence(tn; alg="optimal")
)
  Contract_list = ITensor[]

  for m in mts
    push!(Contract_list, m)
  end

  for v in subgraph
    psiv = flatpsi[v]
    push!(Contract_list, psiv)
  end

  M = contract(Contract_list; sequence=contraction_sequence(Contract_list))

  normalize!(M)

  return M
end

#Do an update of all message tensors for a given flat ITensornetwork and its partition into sub graphs
function update_all_mts(
  flatpsi::ITensorNetwork,
  mts::Dict{Pair,ITensor},
  dg_subgraphs::DataGraph;
  contraction_sequence::Function=tn -> ITensorNetworks.contraction_sequence(tn; alg="optimal")
)
  newmts = Dict{Pair,ITensor}()

  for (key, value) in mts
    mts_to_use = ITensor[]
    subgraph_src = key[1]
    subgraph_dst = key[2]
    connected_subgraphs = neighbors(dg_subgraphs, subgraph_src)
    for k in connected_subgraphs
      if (k != subgraph_dst)
        push!(mts_to_use, mts[k => subgraph_src])
      end
    end
    newmts[subgraph_src => subgraph_dst] = updatemt(
      flatpsi, dg_subgraphs[subgraph_src], mts_to_use; contraction_sequence = contraction_sequence
    )
  end

  return newmts
end

function update_all_mts(
  flatpsi::ITensorNetwork,
  mts::Dict{Pair,ITensor},
  dg_subgraphs::DataGraph,
  niters::Int64;
  contraction_sequence::Function=tn -> ITensorNetworks.contraction_sequence(tn; alg="optimal")
)
  newmts = deepcopy(mts)

  for i in 1:niters
    newmts = update_all_mts(flatpsi, deepcopy(newmts), dg_subgraphs; contraction_sequence = contraction_sequence)
  end

  return newmts
end

#given two flat networks psi and psi0, calculate the ratio of their contraction centred on the the subgraph containing v. The message tensors should be formulated over psi
#Link indices between psi and psi0 should be consistent so the mts can be applied to both
function get_single_site_expec(
  flatpsi::ITensorNetwork,
  flatpsiO::ITensorNetwork,
  mts::Dict{Pair,ITensor},
  dg_subgraphs::DataGraph,
  v::Tuple;
  contraction_sequence::Function=tn -> ITensorNetworks.contraction_sequence(tn; alg="optimal")
)
  es = edges(flatpsi)


  subgraph = find_subgraph(v, dg_subgraphs)
  connected_subgraphs = neighbors(dg_subgraphs, subgraph)
  num_tensors_to_contract = ITensor[]
  denom_tensors_to_contract = ITensor[]
  for k in connected_subgraphs
    push!(num_tensors_to_contract, mts[k => subgraph])
    push!(denom_tensors_to_contract, mts[k => subgraph])
  end

  for vertex in dg_subgraphs[subgraph]
    flatpsiv = flatpsi[vertex]
    flatpsivO = flatpsiO[vertex]
    push!(num_tensors_to_contract, flatpsivO)
    push!(denom_tensors_to_contract, flatpsiv)
  end

  numerator = ITensors.contract(num_tensors_to_contract; sequence=contraction_sequence(num_tensors_to_contract))[1]
  denominator = ITensors.contract(denom_tensors_to_contract; sequence=contraction_sequence(denom_tensors_to_contract))[1]

  return numerator / denominator
end

#given two flat networks psi and psi0, calculate the ratio of their contraction centred on the subgraph(s) containing v1 and v2. The message tensors should be formulated over psi.
function take_2sexpec_two_networks(
  psi::ITensorNetwork,
  psiO::ITensorNetwork,
  mts::Dict{Pair,ITensor},
  dg_subgraphs::DataGraph,
  v1::Tuple,
  v2::Tuple;
  contraction_sequence::Function=tn -> ITensorNetworks.contraction_sequence(tn; alg="optimal")
)
  subgraph1 = find_subgraph(v1, dg_subgraphs)
  subgraph2 = find_subgraph(v2, dg_subgraphs)
  num_tensors_to_contract = ITensor[]
  denom_tensors_to_contract = ITensor[]

  if (subgraph1 == subgraph2)
    connected_subgraphs = neighbors(dg_subgraphs, subgraph1)
    for k in connected_subgraphs
      push!(num_tensors_to_contract, mts[k => subgraph1])
      push!(denom_tensors_to_contract, mts[k => subgraph1])
    end

    for vertex in dg_subgraphs[subgraph1]
      if (vertex != v1 && vertex != v2)
        push!(num_tensors_to_contract, deepcopy(psi[vertex]))
      else
        push!(num_tensors_to_contract, deepcopy(psiO[vertex]))
      end
      push!(denom_tensors_to_contract, deepcopy(psi[vertex]))
    end
  else
    connected_subgraphs1 = neighbors(dg_subgraphs, subgraph1)
    for k in connected_subgraphs1
      if (k != subgraph2)
        push!(num_tensors_to_contract, mts[k => subgraph1])
        push!(denom_tensors_to_contract, mts[k => subgraph1])
      end
    end

    connected_subgraphs2 = neighbors(dg_subgraphs, subgraph2)
    for k in connected_subgraphs2
      if (k != subgraph1)
        push!(num_tensors_to_contract, mts[k => subgraph2])
        push!(denom_tensors_to_contract, mts[k => subgraph2])
      end
    end

    for vertex in dg_subgraphs[subgraph1]
      if (vertex != v1)
        push!(num_tensors_to_contract, deepcopy(psi[vertex]))
      else
        push!(num_tensors_to_contract, deepcopy(psiO[vertex]))
      end

      push!(denom_tensors_to_contract, deepcopy(psi[vertex]))
    end

    for vertex in dg_subgraphs[subgraph2]
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

#Starting with initial guess for messagetensors, monitor the convergence of an observable on a single site v (which is emedded in psiflatO)
function iterate_single_site_expec(
  psiflat::ITensorNetwork,
  psiflatO::ITensorNetwork,
  initmts::Dict{Pair,ITensor},
  dg_subgraphs::DataGraph,
  niters::Int64,
  v::Tuple,
)
  println(
    "Initial Guess for Observable on site " *
    string(v) *
    " is " *
    string(get_single_site_expec(psiflat, psiflatO, initmts, dg_subgraphs, v)),
  )
  mts = deepcopy(initmts)
  for i in 1:niters
    mts = update_all_mts(psiflat, mts, dg_subgraphs, niters)
    approx_O = get_single_site_expec(psiflat, psiflatO, mts, dg_subgraphs, v)
    println(
      "After iteration " *
      string(i) *
      " Belief propagation gives observable on site " *
      string(v) *
      " as " *
      string(approx_O),
    )
  end
end
