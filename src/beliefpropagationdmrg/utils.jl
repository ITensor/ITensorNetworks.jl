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

function smallest_eigvalue(A::AbstractITensorNetwork)
    out = reduce(*, [A[v] for v in vertices(A)])
    out = out * combiner(inds(out; plev = 0))  *combiner(inds(out; plev = 1))
    out = array(out)
    return minimum(real.(eigvals(out)))
end

function bp_renormalize(ψ::ITensorNetwork, ψIψ::BeliefPropagationCache, ψOψs::Vector)

    qf = unpartitioned_graph(partitioned_tensornetwork( ψIψ))
    L = length(vertices(ψ))
    Z = scalar(ψIψ)
    Zval = Z^(-1/(2*L))
    ψ = copy(ψ)
    for v in vertices(ψ)
        form_bra_v, form_ket_v = bra_vertex(qf, v), ket_vertex(qf, v)
        state = ψ[v] * Zval
        state_dag = copy(state)
        state_dag = replaceinds(dag(state_dag), inds(state_dag), dual_index_map(qf).(inds(state_dag)))
        ψ[v] = state
        ψIψ = update_factor(ψIψ, form_ket_v, state)
        ψIψ = update_factor(ψIψ, form_bra_v, state_dag)
        ψOψs = update_factor.(ψOψs, (form_bra_v, ), (state, ))
        ψOψs = update_factor.(ψOψs, (form_ket_v, ), (state_dag, ))
    end

    return ψ, ψIψ, ψOψs
end

function replace_v(bpc::BeliefPropagationCache, v)
    bpc = copy(bpc)
    qf = unpartitioned_graph(partitioned_tensornetwork(bpc))
    v_ket, v_bra, v_op = bra_vertex(qf, v), ket_vertex(qf, v), operator_vertex(qf, v)
    s, sp = commonind(qf[v_op], qf[v_bra]), commonind(qf[v_op], qf[v_ket])
    d = dim(s)
    remaining_inds = intersect(noncommoninds(qf[v_op], qf[v_bra]), noncommoninds(qf[v_op], qf[v_ket]))
    bpc = update_factor(bpc, v_ket, ITensor(sqrt(1.0 / d), inds(qf[v_ket])))
    bpc = update_factor(bpc, v_bra, ITensor(sqrt(1.0 / d), inds(qf[v_bra])))
    bpc = update_factor(bpc, v_op, delta(s,sp)*ITensor(1.0, remaining_inds))
    return bpc
end

function local_op(bpc::BeliefPropagationCache, v)
    qf = unpartitioned_graph(partitioned_tensornetwork(bpc))
    v_op = operator_vertex(qf, v)
    return copy(qf[v_op])
end


function get_local_term(bpc::BeliefPropagationCache, v)
    qf = copy(unpartitioned_graph(partitioned_tensornetwork(bpc)))
    return qf[(v, "bra")]*qf[(v, "ket")]*qf[(v, "operator")]
end