"""Take a vector of gates which act on different edges/ vertices of an Inds network and construct the tno which represents prod(gates)"""
function gate_group_to_tno(s::IndsNetwork, gates::Vector{ITensor}; check_commutativity = true)

   #Construct indsnetwork for TNO
   s_O = copy(s)
   for v in vertices(s_O)
    s_O[v] = Index[s[v]..., s[v]'...]
   end

   if length(gate_groups(s, gates)) != 1
    error("Gates do not all act on different physical degrees of freedom. This is not currently supported.")
   end
   
   O = delta_network(s_O)

   for gate in gates
    v⃗ =  vertices(s)[findall(i -> (length(commoninds(s[i], inds(gate))) != 0), vertices(s))]
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

"""Take a series of gates and convert into a series of tnos"""
function get_tnos(s::IndsNetwork, gates::Vector{ITensor})
    tnos = ITensorNetwork[]
    gate_groups = group_gates(s, gates)
    for group in gate_groups
        push!(tnos, gate_group_to_tno(s, group; check_commutativity = false))
    end

    return tnos
end