function BP_apply(o::ITensor, ψ::AbstractITensorNetwork, bpc::BeliefPropagationCache; apply_kwargs...)
    bpc = copy(bpc)
    ψ = copy(ψ)
    vs = neighbor_vertices(ψ, o)
    envs = environment(bpc, PartitionVertex.(vs))
    singular_values! = Ref(ITensor())
    ψ = noprime(apply(o, ψ; envs, singular_values!, normalize=true, apply_kwargs...))
    ψdag = prime(dag(ψ); sites=[])
    if length(vs) == 2
      v1, v2 = vs
      pe = partitionedge(bpc, (v1, "bra") => (v2, "bra"))
      mts = messages(bpc)
      ind1, ind2 = noncommonind(singular_values![], ψ[v1]), commonind(singular_values![], ψ[v1])
      singular_values![] = denseblocks(replaceind(singular_values![], ind1, ind2'))
      set!(mts, pe, ITensor[singular_values![]])
      set!(mts, reverse(pe), ITensor[singular_values![]])
    end
    for v in vs
      bpc = update_factor(bpc, (v, "ket"), ψ[v])
      bpc = update_factor(bpc, (v, "bra"), ψdag[v])
    end
    return ψ, bpc
end

function smallest_eigvalue(A::AbstractITensorNetwork)
    out = reduce(*, [A[v] for v in vertices(A)])
    out = out * combiner(inds(out; plev = 0))  *combiner(inds(out; plev = 1))
    out = array(out)
    return minimum(real.(eigvals(out)))
end

function heavy_hex_lattice_graph(n::Int64, m::Int64)
    """Create heavy-hex lattice geometry"""
    g = named_hexagonal_lattice_graph(n, m)
    g = decorate_graph_edges(g)

    vertex_rename = Dictionary()
    for (i,v) in enumerate(vertices(g))
        set!(vertex_rename, v, (i,))
    end
    g = rename_vertices(v -> vertex_rename[v], g)
    
    return g
end