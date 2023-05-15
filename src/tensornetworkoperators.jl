"""Given a vector of ITensors, separate them into groups of commuting itensors (i.e. itensors in the same group do not share any common indices)"""
function group_ITensors(its::Vector{ITensor})
  remaining_its = copy(its)
  it_groups = Vector{ITensor}[]

  while !isempty(remaining_its)
    cur_group = ITensor[]
    cur_indices = Index[]
    inds_to_remove = []
    for i in 1:length(remaining_its)
      it = remaining_its[i]
      it_inds = inds(it)

      if all([i ∉ cur_indices for i in it_inds])
        push!(cur_group, it)
        push!(cur_indices, it_inds...)
        push!(inds_to_remove, i)
      end
    end
    remaining_its = ITensor[
      remaining_its[i] for
      i in setdiff([i for i in 1:length(remaining_its)], inds_to_remove)
    ]
    push!(it_groups, cur_group)
  end

  return it_groups
end

"""Take a vector of gates which act on different edges/ vertices of an Inds network and construct the tno which represents prod(gates)"""
function gate_group_to_tno(s::IndsNetwork, gates::Vector{ITensor}; check_commutativity=true)

  #Construct indsnetwork for TNO
  s_O = union_all_inds(s, prime(s; links=[]))

  if check_commutativity && length(group_gates(gates)) != 1
    error(
      "Gates do not all act on different physical degrees of freedom. TNO construction for this is not currently supported.",
    )
  end

  O = delta_network(s_O)

  for gate in gates
    v⃗ = vertices(s)[findall(i -> (length(commoninds(s[i], inds(gate))) != 0), vertices(s))]
    if length(v⃗) == 1
      O[v⃗[1]] = gate
    elseif length(v⃗) == 2
      e = v⃗[1] => v⃗[2]
      if !has_edge(s, e)
        error("Vertices where the gates are being applied must be neighbors for now.")
      end
      Osrc, Odst = factorize(gate, commoninds(O[v⃗[1]], gate))
      O[v⃗[1]] = Osrc
      O[v⃗[2]] = Odst
    else
      error(
        "Can only deal with gates acting on one or two sites for now. Physical indices of the gates must also match those in the IndsNetwork/",
      )
    end
  end

  return O
end

"""Take a series of gates acting on the physical indices specified by IndsNetwork convert into a series of tnos
whose product represents prod(gates)"""
function get_tnos(s::IndsNetwork, gates::Vector{ITensor})
  tnos = ITensorNetwork[]
  gate_groups = group_ITensors(gates)
  for group in gate_groups
    push!(tnos, gate_group_to_tno(s, group; check_commutativity=false))
  end

  return tnos
end
