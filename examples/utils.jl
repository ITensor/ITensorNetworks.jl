using ITensors: siteinds, Op, prime, OpSum, apply
using Graphs: AbstractGraph, SimpleGraph, edges, vertices, is_tree, connected_components
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: decorate_graph_edges, forest_cover, add_edges, rem_edges, add_vertices, rem_vertices, disjoint_union, subgraph, src, dst
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedge, unpartitioned_graph
using ITensorNetworks: BeliefPropagationCache, AbstractITensorNetwork, AbstractFormNetwork, IndsNetwork, ITensorNetwork, insert_linkinds, ttn, union_all_inds,
    neighbor_vertices, environment, messages, update_factor, message, partitioned_tensornetwork, bra_vertex, ket_vertex, operator_vertex, default_cache_update_kwargs,
    dual_index_map
using DataGraphs: underlying_graph
using ITensorNetworks.ModelHamiltonians: heisenberg
using ITensors: ITensor, noprime, dag, noncommonind, commonind, replaceind, dim, noncommoninds, delta, replaceinds
using ITensors.NDTensors: denseblocks
using Dictionaries: set!

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

function heavy_hex_lattice_graph(n::Int64, m::Int64; periodic)
    """Create heavy-hex lattice geometry"""
    g = named_hexagonal_lattice_graph(n, m; periodic)
    g = decorate_graph_edges(g)
    return g
end

function renamer(g)
    vertex_rename = Dictionary()
    for (i,v) in enumerate(vertices(g))
        set!(vertex_rename, v, (i,))
    end
    return rename_vertices(v -> vertex_rename[v], g)
end