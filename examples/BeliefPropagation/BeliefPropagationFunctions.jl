using ITensors
using ITensorNetworks
using NamedGraphs
using DataGraphs
using MultiDimDictionaries
using Graphs
using Random
using LinearAlgebra

maybe_only(x) = x
maybe_only(x::Tuple{T}) where {T} = only(x)

#Function to construct the initial message tensors associated with a given ItensorNetwork (two for each edge in the graph, one forward one backward)
function construct_initial_mts(psi::ITensorNetwork, g::NamedDimGraph, s::IndsNetwork)
    #Make empty lists
    forwardmts = ITensor[]
    backwardmts = ITensor[]
    #RUn over all edges
    for e in edges(g)
        #Get the vitual index in the TensorNetwork which corresponds to that edge
        ind = commoninds(psi, e)[1]
        #Construct two random positive definite matrices and set the forward and backward message tensors to them respectively
        #Normalise them such that their Frobenius norm is 1
        X=rand(dim(ind),dim(ind))
        M=X'*X
        M = M / sqrt(tr(M'*M))
        t1 = ITensor(M, ind, ind')
        X=rand(dim(ind),dim(ind))
        M=X'*X
        M = M / sqrt(tr(M'*M))
        t2 = ITensor(M, ind, ind')

        push!(forwardmts, t1)
        push!(backwardmts, t2)
    end

    return forwardmts, backwardmts
end

#Function to update a message tensor based on Eq. (2) in https://arxiv.org/pdf/2206.04701.pdf
#Take the local tensor for the source node of the edge and contract it with the message tensors (mts) for the edges going into it
function update_mt(psiv::ITensor, mts::Vector{ITensor}, siteind::Index)

    #Form the local tensor
    psidagv = prime(conj(psiv))
    noprime!(psidagv; tags = tags(siteind))
    M = psiv * psidagv
    #Apply all the incoming message tensors to it
    for m in mts
        M = m*M
    end

    #Normalise it
    Mdat = matrix(M)
    M = M / sqrt(tr(Mdat*Mdat'))
    return M
end

#Function to take a TensorNetwork, an initial list of forward and backward message tensors for each edge and update them all based on Eq. (2) in https://arxiv.org/pdf/2206.04701.pdf
function update_all_mts(psi::ITensorNetwork, fmts::Vector{ITensor}, bmts::Vector{ITensor}, g::NamedDimGraph, s::IndsNetwork)
    newfmts = ITensor[]
    newbmts = ITensor[]
    count = 0
    #Run over all the edges in the graph (there is a fmt and bmt for each edge)
    for e in edges(g)
        #Get the source of the edge and the local tensor for that source
        v = maybe_only(src(e))
        psiv = psi[v]
        mts_to_use = ITensor[]
        #Run over all the neighbours of the source (ignoring the destination) and figure out whether the forward or the backward mt is the one to apply
        for vert in neighbors(g, v)
            if(vert != dst(e))
                ind = find_edge(edges(g), vert, src(e))
                if(ind == 0)
                    println("ERROR, CAN'T FIND EDGE ANYWHERE")
                end
                if(ind < 0)
                    push!(mts_to_use, bmts[-ind])
                else
                    push!(mts_to_use, fmts[ind])
                end
            end
            
        end
        #Do Eq. 2 for that edge (forward)
        push!(newfmts, update_mt(psiv, mts_to_use, s[v][1]))


        #Now do the same for the reverse of the edge to get the new bmt for it
        v = maybe_only(dst(e))
        psiv = psi[v]
        mts_to_use = ITensor[]
        for vert in neighbors(g, v)
            if(vert != src(e))
                ind = find_edge(edges(g), vert, dst(e))
                if(ind == 0)
                    println("ERROR, CAN'T FIND EDGE ANYWHERE")
                end
                if(ind < 0)
                    push!(mts_to_use, bmts[-ind])
                else
                    push!(mts_to_use, fmts[ind])
                end
            end
        end
        #Do Eq. 2 for that edge (backward)
        push!(newbmts, update_mt(psiv, mts_to_use, s[v][1]))
        count += 1

    end
    return newfmts, newbmts
end

#Given a list of edges and a source and destination vertex find the index of that edge in the list.
#If the edge has the source and destination reversed then return the negative of the index
function find_edge(edges::Vector{NamedDimEdge{Tuple}}, source::Tuple{Int64}, dest::Tuple{Int64})
    for i = 1:length(edges)
        if(src(edges[i]) == source && dst(edges[i]) == dest)
            return i
        elseif(dst(edges[i]) == source && src(edges[i]) == dest)
            return -i
        end
    end
    return 0
end

#Given an ITensorNetwork associated with a graph g with and inds network s for the message tensors by iterating over an initial guess nbps times
function form_mts(psi::ITensorNetwork, g::NamedDimGraph, s::IndsNetwork, nbps::Int)
    fmts, bmts = construct_initial_mts(psi, g, s)
    for i = 1:nbps
        fmts, bmts = update_all_mts(psi, deepcopy(fmts), deepcopy(bmts), g, s)
    end

    return fmts, bmts
end

#Given an ITensorNetwork associated with a graph g and inds network s with approximate forward and backward message tensors then calculate the local expectation value op
#on every site and return it as a list
function take_local_expec_using_mts(psi::ITensorNetwork, fmts::Vector{ITensor}, bmts::Vector{ITensor}, g::NamedDimGraph, s::IndsNetwork, op::String)
    out = []
    for v in vertices(psi)
        siteind = s[v][1]
        psiv = psi[v]
        psidagv = prime(conj(psiv))
        noprime!(psidagv; tags = tags(siteind))
        O = ITensor(Op(op, v), s)
        c1 = psiv*O
        noprime!(c1)
        c2 = psiv* psidagv
        c1 = c1*psidagv

        for vert in neighbors(g, v)
            ind = find_edge(edges(g), vert, v)
            if(ind == 0)
                println("ERROR, CAN'T FIND EDGE ANYWHERE")
            end
            if(ind < 0)
                c1 = c1*bmts[-ind]
                c2 = c2*bmts[-ind]
            else
                c1 = c1*fmts[ind]
                c2 = c2*fmts[ind]
            end
        end
        push!(out, c1[1]/c2[1])
    end

    return out

end