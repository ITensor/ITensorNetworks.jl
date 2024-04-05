using Graphs: has_edge
using ITensors: ITensors, commoninds, product
using LinearAlgebra: factorize

"""
Take a vector of gates which act on different edges/ vertices of an Inds network and construct the tno which represents prod(gates).
"""
function gate_group_to_tno(s::IndsNetwork, gates::Vector{ITensor})

  #Construct indsnetwork for TNO
  s_O = union_all_inds(s, prime(s; links=[]))

  # Make a TNO with `I` on every site.
  O = ITensorNetwork(Op("I"), s_O)

  for gate in gates
    v⃗ = vertices(s)[findall(i -> (length(commoninds(s[i], inds(gate))) != 0), vertices(s))]
    if length(v⃗) == 1
      O[v⃗[1]] = product(O[v⃗[1]], gate)
    elseif length(v⃗) == 2
      e = v⃗[1] => v⃗[2]
      if !has_edge(s, e)
        error("Vertices where the gates are being applied must be neighbors for now.")
      end
      Osrc, Odst = factorize(gate, commoninds(O[v⃗[1]], gate))
      O[v⃗[1]] = product(O[v⃗[1]], Osrc)
      O[v⃗[2]] = product(O[v⃗[2]], Odst)
    else
      error(
        "Can only deal with gates acting on one or two sites for now. Physical indices of the gates must also match those in the IndsNetwork.",
      )
    end
  end

  return combine_linkinds(O)
end

"""Take a series of gates acting on the physical indices specified by IndsNetwork convert into a series of tnos
whose product represents prod(gates). Useful for keeping the bond dimension of each tno low (as opposed to just building a single tno)"""
function get_tnos(s::IndsNetwork, gates::Vector{ITensor})
  tnos = ITensorNetwork[]
  gate_groups = group_commuting_itensors(gates)
  for group in gate_groups
    push!(tnos, gate_group_to_tno(s, group))
  end

  return tnos
end
