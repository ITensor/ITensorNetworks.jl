using ITensors: siteinds, Op, prime
using Graphs: edges, vertices, is_tree, connected_components
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: forest_cover, add_edges, rem_edges, add_vertices, rem_vertices, disjoint_union, subgraph
using ITensorNetworks: AbstractFormNetwork, IndsNetwork, ITensorNetwork, insert_linkinds, ttn, union_all_inds
using DataGraphs: underlying_graph
using ITensorNetworks.ModelHamiltonians: heisenberg

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

function model_tno(s::IndsNetwork, model::Function; params...)
    forests = forest_cover(s)
    tnos = ITensorNetwork[]
    for s_tree in forests
        g = underlying_graph(s_tree)
        tno = ITensorNetwork(ttn(model(g; params...), s_tree))

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
    end

    return reduce(+, tnos)
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