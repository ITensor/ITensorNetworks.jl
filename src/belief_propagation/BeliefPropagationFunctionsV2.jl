using ITensors
using ITensorNetworks
using NamedGraphs
using DataGraphs
using MultiDimDictionaries
using Graphs
using Random
using LinearAlgebra
using KaHyPar
using Metis

#Belief Propagation but keeping top and bottom separate in the hope for efficiency

#Construct the initial Message Tensors for an ITensor Network, partitioning into npartition subgraphs
#Store only the `top` part of the message tensor, the bottom part is just the conjugate
function GBP_construct_initial_mts(g::NamedDimGraph, psi::ITensorNetwork, npartitions::Int64)

    mts = Dict{Tuple, ITensor}()
    subgraphconns = Dict{Int, Vector{Int}}()
    forwardmts = ITensor[]
    backwardmts = ITensor[]

    #Assign each subgraph to a vertex
    ps = ITensorNetworks.partition(g, npartitions, configuration = :edge_cut, imbalance = 0.0)
    subgraphs = [[v for v in vertices(psi) if ps[v] == s] for s = 1:(npartitions)]

    no_subgraphs = length(subgraphs)
    #Define the subgraph adjacency matrix
    subgraphadjmat = zeros(Int,no_subgraphs, no_subgraphs)

    es = edges(g)
    for i = 1:no_subgraphs
        for j = i+1:no_subgraphs
            sgraph1 = subgraphs[i]
            sgraph2 = subgraphs[j]

            edge_inds = []
            site_indsf = []
            site_indsb = []
            for v1 in sgraph1
                sind = s[v1]
                push!(site_indsf, sind)
                for v2 in sgraph2
                    ind = find_edge(es, v1, v2)
                    if(ind != 0)
                        edge = es[abs(ind)]
                        edge_ind = commoninds(psi, edge)[1]
                        push!(edge_inds, edge_ind)
                    end
                    
                end
            end


            if(!isempty(edge_inds))
                subgraphadjmat[i, j] = 1
                subgraphadjmat[j, i] = 1
                fM12 = normalize!(randomITensor(edge_inds))
                push!(forwardmts, fM12)
                mts[(i,j)] = fM12

                bM12 = normalize!(randomITensor(edge_inds))
                push!(forwardmts, bM12)
                mts[(i,j)] = bM12

            end

        end

        row =  subgraphadjmat[i,:]
        inds = findall(!iszero, row)
        subgraphconns[i] = inds
    end


    #Returns a list of the vertices in each subgraphs, a list of the connecting subgraphes to each subgraph, and the message tensors
    return subgraphs, subgraphconns, mts
end

#DO a single update of a message tensor using the current subgraph and the incoming mts
function GBP_update_mt(g::NamedDimGraph, psi::ITensorNetwork, subgraph::Vector, mts::Vector{ITensor}, s::IndsNetwork)
    Contract_list = ITensor[]

    for m in mts
        push!(Contract_list, m)
    end

    for v in subgraph
        psiv = psi[v]
        push!(Contract_list, psiv)
        sv = s[v][1]
        psidagv = prime(conj(psiv))
        noprime!(psidagv; tags = tags(sv))
        push!(Contract_list, psidagv)
    end

    M = ITensors.contract(Contract_list)

    normalize!(M)

    return M
end

#Do an update of all message tensors for a given ITensornetwork
function GBP_update_all_mts(g::NamedDimGraph, psi::ITensorNetwork, s::IndsNetwork, mts::Dict{Tuple, ITensor}, subgraphs, subgraphconns = Dict{Int, Vector{Int}})
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
        newmts[(subgraph_src, subgraph_dst)] =  GBP_update_mt(g, psi, subgraphs[subgraph_src], mts_to_use, s)
    end

    return newmts
end

#Form the message tensors for a given ITensor Network, using niters iterations and partitioning into npartitions subgraphs
function GBP_form_mts(g::NamedDimGraph, psi::ITensorNetwork, s::IndsNetwork,  niters::Int64, npartitions::Int64)

    subgraphs, subgraphconns, mts = GBP_construct_initial_mts(g, psi, npartitions)

    for i = 1:niters
        mts = GBP_update_all_mts(g, psi, s, deepcopy(mts), subgraphs, subgraphconns)
    end

    return subgraphs, subgraphconns, mts
end

#Calculate a single site expec value from an ITensorNetwork using its mts
function GBP_get_single_site_expec(g::NamedDimGraph, psi::ITensorNetwork, s::IndsNetwork, mts::Dict{Tuple, ITensor}, subgraphs, subgraphconns::Dict{Int, Vector{Int}}, op::String)
    out = Dict{Tuple, Float64}()
    no_subgraphs = length(subgraphs)
    for v in vertices(psi)
        subgraph = find_subgraph(v, subgraphs)
        connected_subgraphs = subgraphconns[subgraph]
        num_tensors_to_contract = ITensor[]
        denom_tensors_to_contract = ITensor[]
        for k in connected_subgraphs
                push!(num_tensors_to_contract, mts[(k, subgraph)])
                push!(denom_tensors_to_contract, mts[(k, subgraph)])
        end

        for vertex in subgraphs[subgraph]
            if(vertex != v)
                sv = s[vertex][1]
                psiv = psi[vertex]
                psidagv = prime(conj(psiv))
                noprime!(psidagv; tags = tags(sv))
                push!(num_tensors_to_contract, psiv, psidagv)
                push!(denom_tensors_to_contract, psiv, psidagv)
            else
                sv = s[vertex][1]
                psiv = psi[vertex]
                psidagv = prime(conj(psiv))
                noprime!(psidagv; tags = tags(sv))
                O = ITensor(Op(op, vertex), s)
                push!(num_tensors_to_contract, noprime!(psiv*O), psidagv)
                push!(denom_tensors_to_contract, psiv, psidagv)
            end
        end
        
        numerator = ITensors.contract(num_tensors_to_contract)[1]
        denominator = ITensors.contract(denom_tensors_to_contract)[1]
        out[v] = numerator/denominator

    end

    return out
end

function GBP_get_two_site_expec(g::NamedDimGraph, psi::ITensorNetwork, s::IndsNetwork, mts::Dict{Tuple, ITensor}, subgraphs, subgraphconns::Dict{Int, Vector{Int}}, op1::String, op2::String, v1 ,v2)
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
            sv = s[vertex][1]
            psiv = psi[vertex]
            psidagv = prime(conj(psiv))
            if(vertex != v1 && vertex != v2)
                noprime!(psidagv; tags = tags(sv))
                push!(num_tensors_to_contract, psiv, psidagv)
            elseif(vertex == v1)
                noprime!(psidagv; tags = tags(sv))
                O = ITensor(Op(op1, vertex), s)
                push!(num_tensors_to_contract, noprime!(psiv*O), psidagv)
            else
                noprime!(psidagv; tags = tags(sv))
                O = ITensor(Op(op2, vertex), s)
                push!(num_tensors_to_contract, noprime!(psiv*O), psidagv)
            end
            push!(denom_tensors_to_contract, psiv, psidagv)
        end
    else
        #IMPLEMENT ERROR MESSAGE IF THE SUBGRAPHS AREN'T ADJACENT!!!
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
                sv = s[vertex][1]
                psiv = psi[vertex]
                psidagv = prime(conj(psiv))
                noprime!(psidagv; tags = tags(sv))
                push!(num_tensors_to_contract, psiv, psidagv)
                push!(denom_tensors_to_contract, psiv, psidagv)
            else
                sv = s[vertex][1]
                psiv = psi[vertex]
                psidagv = prime(conj(psiv))
                noprime!(psidagv; tags = tags(sv))
                O = ITensor(Op(op1, vertex), s)
                push!(num_tensors_to_contract, noprime!(psiv*O), psidagv)
                push!(denom_tensors_to_contract, psiv, psidagv)
            end
        end

        for vertex in subgraphs[subgraph2]
            if(vertex != v2)
                sv = s[vertex][1]
                psiv = psi[vertex]
                psidagv = prime(conj(psiv))
                noprime!(psidagv; tags = tags(sv))
                push!(num_tensors_to_contract, psiv, psidagv)
                push!(denom_tensors_to_contract, psiv, psidagv)
            else
                sv = s[vertex][1]
                psiv = psi[vertex]
                psidagv = prime(conj(psiv))
                noprime!(psidagv; tags = tags(sv))
                O = ITensor(Op(op2, vertex), s)
                push!(num_tensors_to_contract, noprime!(psiv*O), psidagv)
                push!(denom_tensors_to_contract, psiv, psidagv)
            end
        end
    end

    numerator = ITensors.contract(num_tensors_to_contract)[1]
    denominator = ITensors.contract(denom_tensors_to_contract)[1]
    out = numerator/denominator

    return out
end

#Find the subgraph in the list of subgraphs which contains the vertex v 
function find_subgraph(vertex::Tuple, subgraphs)
    for i = 1:length(subgraphs[:,1])
        for s in subgraphs[i, :][1]
            if(s == vertex)
                return i
            end
        end
    end
end

#Find the edge in the list of edges which goes from source to dest (use the negative index if the edge is reversed)
function find_edge(edges::Vector{NamedDimEdge{Tuple}}, source::Tuple, dest::Tuple)
    for i = 1:length(edges)
        if(src(edges[i]) == source && dst(edges[i]) == dest)
            return i
        elseif(dst(edges[i]) == source && src(edges[i]) == dest)
            return -i
        end
    end
    return 0
end

function in_same_subgraph(subgraphs, v1, v2)
    s1 = find_subgraph(v1, subgraphs)
    s2 = find_subgraph(v2, subgraphs)

    if s1 == s2
        return true
    else
        return false
    end
end