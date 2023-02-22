"""Take the square root of an ITensor
Assuming A is PSD, this will achieve the square root"""
function root_ITensor(A::ITensor; leftinds = ())
    U, S, V = svd(A, leftinds...)


    sqrtS = copy(S)
    for i = 1:mindim(S)
        sqrtS[i,i] = sqrt(S[i,i])
    end

    rtA = sqrtS * U
    return rtA

end

"""Take the inverse of an ITensor"""
function inverse_ITensor(A::ITensor; leftinds = ())
    U, S, V = svd(A, leftinds...)

    invS = copy(S)
    for i = 1:mindim(S)
        invS[i,i] = 1.0/S[i,i]
    end

    invA = V*invS*dag(U)

    #A*replaceinds(invA, leftinds, sim(leftinds)) will be identity (contraction over one common index but not the other)

    return invA 

end

"""Put an ITensorNetwork into the symmetric gauge and return the message tensors (which are the diagonal bond matrices)"""
function symmetrise_itensornetwork(ψ::ITensorNetwork)
    ψψ = ψ ⊗ prime(dag(ψ); sites=[])
    vertex_groups = nested_graph_leaf_vertices(partition(ψψ, group(v -> v[1], vertices(ψψ))))
    mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)

    ψsymm = copy(ψ)
    symm_mts = copy(mts)

    for e in edges(ψ)
        vsrc, vdst = src(e), dst(e)
        s1, s2 = find_subgraph((vsrc,1), mts), find_subgraph((vdst,1), mts)
        rootX = root_ITensor(mts[s1 => s2]; leftinds = inds(mts[s1 => s2], plev = 0))
        rootY = root_ITensor(mts[s2 => s1]; leftinds = inds(mts[s2 => s1], plev = 0))
        
        replaceind!(rootX, noncommonind(rootX, mts[s1 => s2]), prime(commonind(rootX, mts[s1 => s2])))
        swapinds!(rootX, inds(rootX, plev = 0), inds(rootX, plev = 1))

        replaceind!(rootY, noncommonind(rootY, mts[s1 => s2]), prime(commonind(rootY, mts[s1 => s2])))
        swapinds!(rootY, inds(rootY, plev = 0), inds(rootY, plev = 1))


        inv_rootX = inverse_ITensor(rootX; leftinds = inds(rootX, plev = 0))
        inv_rootY = inverse_ITensor(rootY; leftinds = inds(rootY, plev = 0))
        swapinds!(inv_rootX, inds(inv_rootX, plev = 0), inds(inv_rootX, plev = 1))
        swapinds!(inv_rootY, inds(inv_rootY, plev = 0), inds(inv_rootY, plev = 1))
        Ce = swapprime(rootX*swapprime(rootY, 0, 2),2,1)
        U, S, V = svd(Ce, inds(Ce, plev = 0))

        rootS = copy(S)
        for i = 1:mindim(S)
            rootS[i,i] = sqrt(S[i,i])
        end

        U = U*rootS
        V = rootS*V
    

        ψsymm[vsrc] = noprime(ψsymm[vsrc]*inv_rootX)
        ψsymm[vdst] = noprime(ψsymm[vdst]*inv_rootY)

        replaceind!(U, commonind(U, S), prime(noncommonind(U, S)))
        swapprime!(replaceind!(V, commonind(V, S), prime(noncommonind(V, S))),1,0)
        ψsymm[vsrc] = noprime(ψsymm[vsrc]*U)
        ψsymm[vdst] = noprime(ψsymm[vdst]*dag(V))

        replaceinds!(S, inds(S), inds(mts[s1 => s2]))
        symm_mts[s1=>s2] = S
        symm_mts[s2=>s1] = S

    end

    return ψsymm, symm_mts

end

"""Put an ITensorNetwork into the symmetric gauge and also return the message tensors (which are the diagonal bond matrices)"""
function symmetrise_itensornetwork_V2(ψ::ITensorNetwork)

    ψψ = ψ ⊗ prime(dag(ψ); sites=[])
    vertex_groups = nested_graph_leaf_vertices(partition(ψψ, group(v -> v[1], vertices(ψψ))))
    mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)

    ψsymm = copy(ψ)
    symm_mts = copy(mts)

    for e in edges(ψ)
        vsrc, vdst = src(e), dst(e)

        s1, s2 = find_subgraph((vsrc,1), mts), find_subgraph((vdst,1), mts)
        edge_ind = commoninds(mts[s1 => s2], ψsymm[vsrc])
        edge_ind_sim = sim(edge_ind)
        ψsymm[vdst] = replaceinds(ψsymm[vdst], edge_ind, edge_ind_sim)

        rootX = root_ITensor(mts[s1 => s2]; leftinds = edge_ind)
        rootY = root_ITensor(mts[s2 => s1]; leftinds = edge_ind)

        bond_ind = uniqueinds(rootX, rootY)

        replaceinds!(rootY, uniqueinds(rootY, rootX), bond_ind)
        replaceinds!(rootY, edge_ind, edge_ind_sim)

        inv_rootX = inverse_ITensor(rootX; leftinds = edge_ind)
        inv_rootY = inverse_ITensor(rootY; leftinds = edge_ind_sim)

        ψsymm[vsrc] = replaceinds(ψsymm[vsrc]*inv_rootX, bond_ind, edge_ind)
        ψsymm[vdst] = replaceinds(ψsymm[vdst]*inv_rootY, bond_ind, edge_ind_sim)


        Ce = swapinds(rootX, bond_ind, edge_ind)*swapinds(rootY, bond_ind, edge_ind_sim)
        U, S, V = svd(Ce, edge_ind)

        rootS = copy(S)
        for i = 1:mindim(S)
            rootS[i,i] = sqrt(S[i,i])
        end

        ψsymm[vsrc] = replaceinds(ψsymm[vsrc]*U*rootS, commoninds(S, V), edge_ind) 
        ψsymm[vdst] = replaceinds(ψsymm[vdst]*dag(rootS*V), commoninds(U,S), edge_ind)

        replaceinds!(S, inds(S), inds(mts[s1 => s2]))
        symm_mts[s1=>s2] = S
        symm_mts[s2=>s1] = S

    end

    return ψsymm, symm_mts

end