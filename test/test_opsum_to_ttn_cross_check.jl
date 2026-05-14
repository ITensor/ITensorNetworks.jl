using DataGraphs: vertex_data
using Dictionaries: Dictionary
using Graphs: add_edge!, add_vertex!, rem_edge!, vertices
using ITensorNetworks: ITensorNetworks, TreeTensorNetwork, siteinds
using ITensors.NDTensors: with_auto_fermion
using ITensors: @disable_warn_order, Index, contract, removeqns
using LinearAlgebra: norm
using NamedGraphs.GraphsExtensions: leaf_vertices, post_order_dfs_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_path_graph
using Test: @test, @testset
include("utils.jl")
using .ModelHamiltonians: ModelHamiltonians

# Cross-check `TreeTensorNetwork(os::OpSum, sites::IndsNetwork)` on a comb tree
# against the same constructor on a linearized path graph that carries the same
# site indices. Both must produce equivalent dense Hamiltonians: the OpSum and
# its site indices are unchanged, only the auxiliary tree topology differs.

# Path-graph IndsNetwork carrying the supplied `sites` in vertex order
# `1:length(sites)`. `siteinds(f, g)` walks `vertices(g)` and assigns
# `to_siteind(f(v), v)`, which preserves an existing `Index` value unchanged.
function path_siteinds(sites::Vector{<:Index})
    g = named_path_graph(length(sites))
    return siteinds(v -> sites[v], g)
end

@testset "OpSum to TTN: comb tree vs path linearization" begin
    @testset "OpSum to TTN" begin
        # small comb tree
        tooth_lengths = fill(2, 3)
        c = named_comb_tree(tooth_lengths)

        is = siteinds("S=1/2", c)

        # linearized version
        linear_order = [4, 1, 2, 5, 3, 6]
        vmap = Dictionary(collect(vertices(is))[linear_order], eachindex(linear_order))
        sites = only.(collect(vertex_data(is)))[linear_order]
        is_line = path_siteinds(sites)

        # test with next-to-nearest-neighbor Ising Hamiltonian
        J1 = -1
        J2 = 2
        h = 0.5
        H = ModelHamiltonians.ising(c; J1 = J1, J2 = J2, h = h)
        # add combination of longer range interactions
        Hlr = copy(H)
        Hlr += 5, "Z", (1, 2), "Z", (2, 2), "Z", (3, 2)
        Hlr += -4, "Z", (1, 1), "Z", (2, 2), "Z", (3, 1)
        Hlr += 2.0, "Z", (2, 2), "Z", (3, 2)
        Hlr += -1.0, "Z", (1, 2), "Z", (3, 1)

        Hline =
            TreeTensorNetwork(replace_vertices(v -> vmap[v], H), is_line; cutoff = 1.0e-10)
        Hline_lr = TreeTensorNetwork(
            replace_vertices(v -> vmap[v], Hlr), is_line; cutoff = 1.0e-10
        )

        @testset "Svd approach" for root_vertex in leaf_vertices(is)
            Hsvd = TreeTensorNetwork(H, is; root_vertex, cutoff = 1.0e-10)
            @disable_warn_order begin
                @test contract(Hsvd) ≈ contract(Hline) rtol = 1.0e-6
            end

            Hsvd_lr = TreeTensorNetwork(Hlr, is; root_vertex, cutoff = 1.0e-10)
            @disable_warn_order begin
                @test contract(Hsvd_lr) ≈ contract(Hline_lr) rtol = 1.0e-6
            end
        end
    end

    @testset "OpSum to TTN QN" begin
        # small comb tree
        tooth_lengths = fill(2, 3)
        c = named_comb_tree(tooth_lengths)
        is = siteinds("S=1/2", c; conserve_qns = true)
        is_noqns = copy(is)
        for v in vertices(is)
            is_noqns[v] = removeqns(is_noqns[v])
        end

        # linearized version
        linear_order = [4, 1, 2, 5, 3, 6]
        vmap = Dictionary(collect(vertices(is))[linear_order], eachindex(linear_order))
        sites = only.(collect(vertex_data(is)))[linear_order]
        is_line = path_siteinds(sites)

        # test with next-to-nearest-neighbor Heisenberg Hamiltonian
        J1 = -1
        J2 = 2
        h = 0.5
        H = ModelHamiltonians.heisenberg(c; J1 = J1, J2 = J2, h = h)
        # add combination of longer range interactions
        Hlr = copy(H)
        Hlr += 5, "Z", (1, 2), "Z", (2, 2)
        Hlr += -4, "Z", (1, 1), "Z", (2, 2)
        Hlr += 2.0, "Z", (2, 2), "Z", (3, 2)
        Hlr += -1.0, "Z", (1, 2), "Z", (3, 1)

        Hline =
            TreeTensorNetwork(replace_vertices(v -> vmap[v], H), is_line; cutoff = 1.0e-10)
        Hline_lr = TreeTensorNetwork(
            replace_vertices(v -> vmap[v], Hlr), is_line; cutoff = 1.0e-10
        )

        @testset "Svd approach" for root_vertex in leaf_vertices(is)
            Hsvd = TreeTensorNetwork(H, is; root_vertex, cutoff = 1.0e-10)
            @disable_warn_order begin
                @test contract(Hsvd) ≈ contract(Hline) rtol = 1.0e-6
            end

            Hsvd_lr = TreeTensorNetwork(Hlr, is; root_vertex, cutoff = 1.0e-10)
            @disable_warn_order begin
                @test contract(Hsvd_lr) ≈ contract(Hline_lr) rtol = 1.0e-6
            end
        end
    end

    @testset "OpSum to TTN Fermions" begin
        with_auto_fermion() do
            # small comb tree
            tooth_lengths = fill(2, 3)
            c = named_comb_tree(tooth_lengths)
            is = siteinds("Fermion", c; conserve_nf = true)

            # test with next-nearest neighbor tight-binding model
            t = 1.0
            tp = 0.4
            U = 0.0
            h = 0.5
            H = ModelHamiltonians.tight_binding(c; t, tp, h)

            @testset "Svd approach" for root_vertex in leaf_vertices(is)
                # get TTN Hamiltonian on the comb tree directly
                Hsvd = TreeTensorNetwork(H, is; root_vertex, cutoff = 1.0e-10)

                # linearize along the root's post-order DFS and build the same
                # OpSum as a path-graph TTN with the matching site assignment.
                vorder = reverse(post_order_dfs_vertices(c, root_vertex))
                sites = [only(is[v]) for v in vorder]
                vmap = Dictionary(vorder, 1:length(sites))
                is_line = path_siteinds(sites)
                Hline = TreeTensorNetwork(
                    replace_vertices(v -> vmap[v], H), is_line; cutoff = 1.0e-10
                )

                @disable_warn_order begin
                    Tline = contract(Hline)
                    Tttno = contract(Hsvd)
                end

                @test norm(Tline) > 0
                @test norm(Tttno) > 0
                @test Tline ≈ Tttno rtol = 1.0e-6
            end
        end
    end

    @testset "OpSum to TTN QN missing" begin
        # small comb tree
        tooth_lengths = fill(2, 3)
        c = named_comb_tree(tooth_lengths)
        c2 = copy(c)
        ## add an internal vertex into the comb graph c2
        add_vertex!(c2, (-1, 1))
        add_edge!(c2, (-1, 1) => (2, 2))
        add_edge!(c2, (-1, 1) => (3, 1))
        add_edge!(c2, (-1, 1) => (2, 1))
        rem_edge!(c2, (2, 1) => (2, 2))
        rem_edge!(c2, (2, 1) => (3, 1))

        is = siteinds("S=1/2", c; conserve_qns = true)
        is_missing_site = siteinds("S=1/2", c2; conserve_qns = true)
        is_missing_site[(-1, 1)] = Vector{Index}[]

        # linearized version
        linear_order = [4, 1, 2, 5, 3, 6]
        vmap = Dictionary(collect(vertices(is))[linear_order], eachindex(linear_order))
        sites =
            only.(filter(d -> !isempty(d), collect(vertex_data(is_missing_site))))[linear_order]
        is_line = path_siteinds(sites)

        J1 = -1
        J2 = 2
        h = 0.5
        # connectivity of the Hamiltonian is that of the original comb graph
        H = ModelHamiltonians.heisenberg(c; J1, J2, h)

        # add combination of longer range interactions
        Hlr = copy(H)
        Hlr += 5, "Z", (1, 2), "Z", (2, 2)
        Hlr += -4, "Z", (1, 1), "Z", (2, 2)
        Hlr += 2.0, "Z", (2, 2), "Z", (3, 2)
        Hlr += -1.0, "Z", (1, 2), "Z", (3, 1)

        Hline =
            TreeTensorNetwork(replace_vertices(v -> vmap[v], H), is_line; cutoff = 1.0e-10)
        Hline_lr = TreeTensorNetwork(
            replace_vertices(v -> vmap[v], Hlr), is_line; cutoff = 1.0e-10
        )

        @testset "Svd approach" for root_vertex in leaf_vertices(is)
            Hsvd = TreeTensorNetwork(H, is_missing_site; root_vertex, cutoff = 1.0e-10)
            @disable_warn_order begin
                @test contract(Hsvd) ≈ contract(Hline) rtol = 1.0e-6
            end

            Hsvd_lr = TreeTensorNetwork(Hlr, is_missing_site; root_vertex, cutoff = 1.0e-10)
            @disable_warn_order begin
                @test contract(Hsvd_lr) ≈ contract(Hline_lr) rtol = 1.0e-6
            end
        end
    end
end
