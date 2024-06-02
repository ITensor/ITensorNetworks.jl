using ITensors: siteinds, Op, prime, OpSum, apply
using Graphs: AbstractGraph, SimpleGraph, edges, vertices, is_tree, connected_components
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: decorate_graph_edges, forest_cover, add_edges, rem_edges, add_vertices, rem_vertices, disjoint_union, subgraph, src, dst
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedge
using ITensorNetworks: BeliefPropagationCache, AbstractITensorNetwork, AbstractFormNetwork, IndsNetwork, ITensorNetwork, insert_linkinds, ttn, union_all_inds,
    neighbor_vertices, environment, messages, update_factor, message
using DataGraphs: underlying_graph
using ITensorNetworks.ModelHamiltonians: heisenberg
using ITensors: ITensor, noprime, dag, noncommonind, commonind, replaceind
using ITensors.NDTensors: denseblocks

function NamedGraphs.GraphsExtensions.forest_cover(s::IndsNetwork; kwargs...)
    g = underlying_graph(s)
    ss = IndsNetwork[]
    no_edges=  0
    for f in forest_cover(g)
        for g_tree_vertices in connected_components(f)
            g_tree = f[g_tree_vertices]
            if(length(edges(g_tree))) != 0
                s_f = subgraph(v -> v ∈ vertices(g_tree), s)
                edges_to_remove = filter(e -> e ∉ edges(g_tree) && reverse(e) ∉ edges(g_tree), edges(s_f))
                s_f = rem_edges(s_f, edges_to_remove)
                push!(ss, s_f)
                @assert is_tree(s_f)
                no_edges += length(edges(s_f))
            end
        end
    end

    @assert no_edges == length(edges(s))

    return ss
end

function model_tno(s::IndsNetwork, model::Function; two_site_params, one_site_params)
    forests = forest_cover(s)
    tnos = ITensorNetwork[]
    n_forests = length(forests)
    modified_one_site_params = deepcopy(one_site_params)
    for s_tree in forests
        g = underlying_graph(s_tree)
        tno = ITensorNetwork(ttn(model(g; two_site_params..., modified_one_site_params...), s_tree))

        missing_vertices = setdiff(vertices(s), vertices(g))
        s_remaining_verts = subgraph(v -> v ∈ missing_vertices, s)

        @assert issubset(edges(s_tree), edges(s))
        @assert issubset(edges(s_remaining_verts), edges(s))
        identity_tno = ITensorNetwork(Op("I"), union_all_inds(s_remaining_verts, prime(s_remaining_verts)))
        tno = disjoint_union(tno, identity_tno)
        tno = rename_vertices(v -> first(v), tno)
        missing_edges = filter(e -> e ∉ edges(tno) && reverse(e) ∉ edges(tno), edges(s))
        missing_edges = NamedEdge[e ∈ edges(s) ? e : reverse(e) for e in missing_edges]
        tno = insert_linkinds(tno, missing_edges; link_space = 1)
        push!(tnos, tno)

        for param in modified_one_site_params
            for v in vertices(g)
                if haskey(param, v)
                    delete!(param ,v)
                end
            end
        end
    end

    return reduce(+, tnos)
end

function model_tno_simplified(s::IndsNetwork, model::Function; two_site_params, one_site_params)
    forests = forest_cover(s)
    tnos = ITensorNetwork[]
    n_forests = length(forests)
    modified_one_site_params = deepcopy(one_site_params)
    for s_tree in forests
        g = underlying_graph(s_tree)
        tno = ITensorNetwork(ttn(model(g; two_site_params..., modified_one_site_params...), s_tree))

        missing_vertices = setdiff(vertices(s), vertices(g))
        s_remaining_verts = subgraph(v -> v ∈ missing_vertices, s)

        @assert issubset(edges(s_tree), edges(s))
        @assert issubset(edges(s_remaining_verts), edges(s))
        identity_tno = ITensorNetwork(Op("I"), union_all_inds(s_remaining_verts, prime(s_remaining_verts)))
        tno = disjoint_union(tno, identity_tno)
        tno = rename_vertices(v -> first(v), tno)
        push!(tnos, tno)

        for param in modified_one_site_params
            for v in vertices(g)
                if haskey(param, v)
                    delete!(param ,v)
                end
            end
        end
    end

    return tnos
end

function ising_dictified(g::AbstractGraph; Js, hs, hls)
    ℋ = OpSum()
    for e in edges(g)
      if haskey(Js, e) && !iszero(Js[e])
        ℋ += Js[e], "Sz", src(e), "Sz", dst(e)
      end
    end
    for v in vertices(g)
      if haskey(hs, v) && !iszero(hs[v])
        ℋ += hs[v], "Sx", v
      end
      if haskey(hls, v) && !iszero(hls[v])
        ℋ += hls[v], "Sz", v
      end
    end
    return ℋ
end

function xyz_dictified(g::AbstractGraph; Jxs, Jys, Jzs)
    ℋ = OpSum()
    for e in edges(g)
      ℋ += Jxs[e], "X", src(e), "X", dst(e)
      ℋ += Jys[e], "Y", src(e), "Y", dst(e)
      ℋ += Jzs[e], "Z", src(e), "Z", dst(e)
    end
    return ℋ
end

function n_site_expect(ψIψ::AbstractFormNetwork, ops::Vector{String}, vs; kwargs...)
    ψIψ_vs = [ψIψ[operator_vertex(ψIψ, v)] for v in vs]
    s_vs = [commonind(ψIψ[ket_vertex(ψIψ, v)], ψIψ_vs[i]) for (i, v) in enumerate(vs)]
    operators = [ITensors.op(ops[i].which_op, s_vs[i])  for i in 1:length(ops)]
    ∂ψIψ_∂vs = environment(ψIψ, operator_vertices(ψIψ, [vs]); kwargs...)
    numerator = contract([∂ψIψ_∂vs; operators]; contract_kwargs...)[]
    denominator = contract([∂ψIψ_∂vs; ψIψ_vs]; contract_kwargs...)[]
  
    return numerator / denominator
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