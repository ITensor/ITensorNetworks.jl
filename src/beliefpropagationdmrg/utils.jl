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

function exact_energy(g::AbstractGraph, bpc::BeliefPropagationCache)
    tn = ITensorNetwork(g)
    for v in vertices(g)
        tn[v] =  get_local_term(bpc, v)
    end
    degree_two_sites = filter(v -> degree(tn, v) == 2, vertices(tn))
    while !isempty(degree_two_sites)
        v = first(degree_two_sites)
        vn = first(neighbors(g, v))
        tn = contract(tn, NamedEdge(v => vn); merged_vertex=vn)
        degree_two_sites = filter(v -> degree(tn, v) == 2, vertices(tn))
    end
    return ITensors.contract(ITensor[tn[v] for v in vertices(tn)]; sequence = "automatic")[]
end
