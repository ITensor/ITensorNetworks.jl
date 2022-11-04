#Construct the random initial Message Tensors for an ITensor Network, based on a partitioning into subgraphs specified ny 'sub graphs'
#The ITensorNetwork needs to be flat (i.e. just sites and link indices, no site indices)
#If combiners are sent through then get mts by tracing out bonds in original network
#Or else: id_init = 0 => Random, id_init = 1 => Identity Matrix initialisation (preferred)
function construct_initial_mts(g::NamedDimGraph, flatpsi::ITensorNetwork, s::IndsNetwork, subgraphs::Dict{Int, Vector{Tuple}}, subgraphconns::Dict{Int, Vector{Int}}; combiners =Dict{NamedDimEdge{Tuple}, ITensor}(), id_init = 0)

    mts = Dict{Tuple, ITensor}()
    no_cs = length(keys(combiners))

    no_subgraphs = length(subgraphs)
    es = edges(g)
    for i = 1:no_subgraphs
        tns_to_contract = ITensor[]
        for j in subgraphconns[i]
            edge_inds = []

            if(no_cs == 0)
                for vertex in subgraphs[i]
                    psiv = flatpsi[vertex]
                    es_v = find_edges_involving_vertex(es, vertex)
                    for e in es_v
                        if(find_subgraph(dst(e), subgraphs) == j || find_subgraph(src(e), subgraphs) == j)
                            edge_ind = commoninds(flatpsi, e)[1]
                            push!(edge_inds, edge_ind)
                        end
                    end
                end
                if(id_init == 1)
                    A = Array(delta(edge_inds), edge_inds)
                    X1 = ITensor(A, edge_inds)
                else
                    X1 = randomITensor(edge_inds)
                    normalize!(X1)
                end
                mts[(i,j)] = X1
            else
                for vertex in subgraphs[i]
                    psiv = flatpsi[vertex]
                    es_v = find_edges_involving_vertex(es, vertex)
                    for e in es_v
                        if(find_subgraph(dst(e), subgraphs) != j && find_subgraph(src(e), subgraphs) != j)
                            C = combiners[e]
                            psiv = dag(C)*psiv
                            is = commoninds(C, psiv)
                            d = delta(is)
                            psiv = d*psiv
                        end
                    end
                    push!(tns_to_contract, psiv)
    
                end
                mts[(i,j)] = ITensors.contract(tns_to_contract)
            end
        end
    end

    return mts
end

#DO a single update of a message tensor using the current subgraph and the incoming mts
function updatemt(g::NamedDimGraph, flatpsi::ITensorNetwork, subgraph::Vector{Tuple}, mts::Vector{ITensor}, s::IndsNetwork)
    Contract_list = ITensor[]

    for m in mts
        push!(Contract_list, m)
    end

    for v in subgraph
        psiv = flatpsi[v]
        push!(Contract_list, psiv)
    end

    M = ITensors.contract(Contract_list)

    normalize!(M)

    return M
end

#Do an update of all message tensors for a given flat ITensornetwork and its partition into sub graphs
function update_all_mts(g::NamedDimGraph, flatpsi::ITensorNetwork, s::IndsNetwork, mts::Dict{Tuple, ITensor}, subgraphs::Dict{Int, Vector{Tuple}}, subgraphconns::Dict{Int, Vector{Int}})
    newmts =Dict{Tuple, ITensor}()

    for (key, value) in mts
        mts_to_use = ITensor[]
        subgraph_src =key[1]
        subgraph_dst = key[2]
        connected_subgraphs = subgraphconns[subgraph_src]
        for k in connected_subgraphs
            if(k != subgraph_dst)
                push!(mts_to_use, mts[(k, subgraph_src)])
            end
        end
        newmts[(subgraph_src, subgraph_dst)] =  updatemt(g, flatpsi, subgraphs[subgraph_src], mts_to_use, s)
    end

    return newmts
end

function update_all_mts(g::NamedDimGraph, flatpsi::ITensorNetwork, s::IndsNetwork, mts::Dict{Tuple, ITensor}, subgraphs::Dict{Int, Vector{Tuple}}, subgraphconns::Dict{Int, Vector{Int}}, niters::Int64)

    newmts = deepcopy(mts)

    for i = 1:niters
        newmts = update_all_mts(g, flatpsi, s, deepcopy(newmts), subgraphs, subgraphconns)
    end

    return newmts
end

#given two flat networks psi and psi0, calculate the ratio of their contraction centred on the the subgraph containing v. The message tensors should be formulated over psi
#Link indices between psi and psi0 should be consistent so the mts can be applied to both
function get_single_site_expec(g::NamedDimGraph, flatpsi::ITensorNetwork, flatpsiO::ITensorNetwork, s::IndsNetwork, mts::Dict{Tuple, ITensor}, subgraphs::Dict{Int, Vector{Tuple}}, subgraphconns::Dict{Int, Vector{Int}}, v::Tuple)
    no_subgraphs = length(subgraphs)
    es = edges(flatpsi)

    subgraph = find_subgraph(v, subgraphs)
    connected_subgraphs = subgraphconns[subgraph]
    num_tensors_to_contract = ITensor[]
    denom_tensors_to_contract = ITensor[]
    for k in connected_subgraphs
            push!(num_tensors_to_contract, mts[(k, subgraph)])
            push!(denom_tensors_to_contract, mts[(k, subgraph)])
    end


    for vertex in subgraphs[subgraph]
        sv = s[vertex][1]
        flatpsiv = flatpsi[vertex]
        flatpsivO = flatpsiO[vertex]
        push!(num_tensors_to_contract, flatpsivO)
        push!(denom_tensors_to_contract, flatpsiv)
    end


    numerator = ITensors.contract(num_tensors_to_contract)[1]
    denominator = ITensors.contract(denom_tensors_to_contract)[1]


    return  numerator/denominator
end

#given two flat networks psi and psi0, calculate the ratio of their contraction centred on the the subgraph(s) containing v1 and v2. The message tensors should be formulated over psi.
function take_2sexpec_two_networks(g::NamedDimGraph, psi::ITensorNetwork, psiO::ITensorNetwork, s::IndsNetwork, mts::Dict{Tuple, ITensor}, subgraphs::Dict{Int, Vector{Tuple}}, subgraphconns::Dict{Int, Vector{Int}}, v1::Tuple, v2::Tuple)
    subgraph1 = find_subgraph(v1, subgraphs)
    subgraph2 = find_subgraph(v2, subgraphs)
    num_tensors_to_contract = ITensor[]
    denom_tensors_to_contract = ITensor[]

    if(subgraph1 == subgraph2)
        connected_subgraphs = subgraphconns[subgraph1]
        for k in connected_subgraphs
                push!(num_tensors_to_contract, mts[(k, subgraph1)])
                push!(denom_tensors_to_contract, mts[(k, subgraph1)])
        end

        for vertex in subgraphs[subgraph1]
            if(vertex != v1 && vertex != v2)
                push!(num_tensors_to_contract, deepcopy(psi[vertex]))
            else
                push!(num_tensors_to_contract, deepcopy(psiO[vertex]))
            end
            push!(denom_tensors_to_contract, deepcopy(psi[vertex]))
        end
    else
        connected_subgraphs1 = subgraphconns[subgraph1]
        for k in connected_subgraphs1
            if(k != subgraph2)
                push!(num_tensors_to_contract, mts[(k, subgraph1)])
                push!(denom_tensors_to_contract, mts[(k, subgraph1)])
            end
        end
        
        connected_subgraphs2 = subgraphconns[subgraph2]
        for k in connected_subgraphs2
            if(k != subgraph1)
                push!(num_tensors_to_contract, mts[(k, subgraph2)])
                push!(denom_tensors_to_contract, mts[(k, subgraph2)])
            end
        end

        for vertex in subgraphs[subgraph1]
            if(vertex != v1)
                push!(num_tensors_to_contract, deepcopy(psi[vertex]))
            else
                push!(num_tensors_to_contract, deepcopy(psiO[vertex]))
            end

            push!(denom_tensors_to_contract,  deepcopy(psi[vertex]))
        end

        for vertex in subgraphs[subgraph2]
            if(vertex != v2)
                push!(num_tensors_to_contract, deepcopy(psi[vertex]))
            else
                push!(num_tensors_to_contract, deepcopy(psiO[vertex]))
            end

            push!(denom_tensors_to_contract,  deepcopy(psi[vertex]))
        end
    end

    numerator = ITensors.contract(num_tensors_to_contract)[1]
    denominator = ITensors.contract(denom_tensors_to_contract)[1]
    out = numerator/denominator

    return out
end
