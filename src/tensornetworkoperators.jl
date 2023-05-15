"""Given a vector of gates acting on siteinds within s, separate them into groups of commuting gates (i.e. gates in the same group act on different physical indices)"""
function group_gates(s::IndsNetwork, gates::Vector{ITensor})
  remaining_gates = copy(gates)
  gate_groups = Vector{ITensor}[]

  while !isempty(remaining_gates)
    cur_group = ITensor[]
    cur_vertices = []
    inds_to_remove = []
    for i in 1:length(remaining_gates)
      gate = remaining_gates[i]
      vs = vertices(s)[findall(
        i -> (length(commoninds(s[i], inds(gate))) != 0), vertices(s)
      )]

      if isempty(vs)
        error("Gate does not appear to have any indices within the indsnetwork provided")
      end

      if all([v ∉ cur_vertices for v in vs])
        push!(cur_group, gate)
        push!(cur_vertices, vs...)
        push!(inds_to_remove, i)
      end
    end
    remaining_gates = ITensor[
      remaining_gates[i] for
      i in setdiff([i for i in 1:length(remaining_gates)], inds_to_remove)
    ]
    push!(gate_groups, cur_group)
  end

  return gate_groups
end

"""Take a vector of gates which act on different edges/ vertices of an Inds network and construct the tno which represents prod(gates)"""
function gate_group_to_tno(s::IndsNetwork, gates::Vector{ITensor}; check_commutativity=true)

  #Construct indsnetwork for TNO
  s_O = union_all_inds(s, prime(s; links=[]))

  if check_commutativity && length(group_gates(s, gates)) != 1
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
      error("Can only deal with gates acting on one or two sites for now.")
    end
  end

  return O
end

"""Take a series of gates acting on the physical indices specified by IndsNetwork convert into a series of tnos
whose product represents prod(gates)"""
function get_tnos(s::IndsNetwork, gates::Vector{ITensor})
  tnos = ITensorNetwork[]
  gate_groups = group_gates(s, gates)
  for group in gate_groups
    push!(tnos, gate_group_to_tno(s, group; check_commutativity=false))
  end

  return tnos
end
