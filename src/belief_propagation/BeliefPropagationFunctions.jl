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


function GBP_construct_initial_mts(g::NamedDimGraph, psi::ITensorNetwork, npartitions::Int64)

    mts = Dict{Tuple, ITensor}()
    subgraphconns = Dict{Int, Vector{Int}}()
    forwardmts = ITensor[]
    backwardmts = ITensor[]

    #Assign each subgraph to a vertex
    ps = ITensorNetworks.partition(g, npartitions, configuration = :edge_cut, imbalance = 0.0)
    subgraphs = [[v for v in vertices(psi) if ps[v] == s] for s = 1:(npartitions)]
    display(subgraphs)

    no_subgraphs = length(subgraphs)
    #Define the subgraph adjacency matrix
    subgraphadjmat = zeros(Int,no_subgraphs, no_subgraphs)

    es = edges(g)
    for i = 1:no_subgraphs
        for j = i+1:no_subgraphs
            sgraph1 = subgraphs[i]
            sgraph2 = subgraphs[j]

            edge_inds = []
            for v1 in sgraph1
                for v2 in sgraph2
                    ind = find_edge(es, v1, v2)
                    if(ind != 0)
                        edge = es[abs(ind)]
                        edge_ind = commoninds(psi, edge)[1]
                        push!(edge_inds, edge_ind, edge_ind')
                    end
                end
            end
            if(!isempty(edge_inds))
                subgraphadjmat[i, j] = 1
                subgraphadjmat[j, i] = 1
                X1 = randomITensor(edge_inds)
                X2 = prime(dag(X1))
                fM12 = X1*X2
                normalize!(fM12)
                swapprime!(fM12, 2=>1)
                push!(forwardmts, fM12)
                mts[(i,j)] = fM12

                X1 = randomITensor(edge_inds)
                X2 = prime(dag(X1))
                bM12 = X1*X2
                normalize!(bM12)
                swapprime!(bM12, 2=>1)
                push!(backwardmts, bM12)
                mts[(j,i)] = bM12

            end

        end

        row =  subgraphadjmat[i,:]
        inds = findall(!iszero, row)
        subgraphconns[i] = inds
    end


    return subgraphs, subgraphconns, mts
end

function GBP_update_mt(g::NamedDimGraph, psi::ITensorNetwork, subgraph::Vector, mts::Vector{ITensor}, s::IndsNetwork)
    Contract_list = ITensor[]

    for m in mts
        push!(Contract_list, m)
    end

    for v in subgraph
        sv = s[v][1]
        psiv = psi[v]
        psidagv = prime(conj(psiv))
        noprime!(psidagv; tags = tags(sv))
        push!(Contract_list, psiv)
        push!(Contract_list, psidagv)
    end

    M = ITensors.contract(Contract_list)

    normalize!(M)

    return M
end

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

function GBP_form_mts(g::NamedDimGraph, psi::ITensorNetwork, s::IndsNetwork,  niters::Int64, npartitions::Int64)

    subgraphs, subgraphconns, mts = GBP_construct_initial_mts(g, psi, npartitions)

    for i = 1:niters
        mts = GBP_update_all_mts(g, psi, s, deepcopy(mts), subgraphs, subgraphconns)
    end

    return subgraphs, subgraphconns, mts
end

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

function find_subgraph(v::Tuple, subgraphs)
    for i = 1:length(subgraphs[:,1])
        for s in subgraphs[i, :][1]
            if(s == v)
                return i
            end
        end
    end
end

function graph_tensor_network(s, g::NamedDimGraph, beta::Float64; link_space)
    ψ = ITensorNetwork(s; link_space)
    J = 1
    f1 = 0.5*sqrt(exp(-J*beta*0.5)*(-1+exp(J*beta)))
    f2 = 0.5*sqrt(exp(-J*beta*0.5)*(1+exp(J*beta)))
    A = [(f1 + f2)  (-f1 + f2); (-f1 + f2) (f1 + f2)]
    for v in vertices(ψ)
      is = inds(ψ[v])
      ψ[v] = delta(is)
      indices = inds(ψ[v])
      for i in indices
        if(!hastags(i,"Site"))
            Atens = ITensor(A, i, i')
            ψ[v] = ψ[v]*Atens
            ψ[v] = noprime!(ψ[v])
        end
      end

    end
  
    return ψ
end
