using Compat: Compat
using Dictionaries: Dictionary, set!
using Graphs: dst, src, vertices
using ITensorNetworks: ITensorNetworks, BeliefPropagationCache, QuadraticFormNetwork,
    contract, contraction_sequence, environment, identity_messages, inner_network, message,
    message_diff, partitioned_tensornetwork, scalar, siteinds, tensornetwork, update,
    update_factor, updated_message, ⊗
include("utils.jl")
using ITensors.NDTensors: array
using ITensors: ITensors, Algorithm, ITensor, Index, combiner, commoninds, dag, inds, inner,
    op, prime, random_itensor
using LinearAlgebra: eigvals, tr
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NamedGraphs.PartitionedGraphs: PartitionedGraph, QuotientEdge, quotientedges
using NamedGraphs: NamedEdge, NamedGraph
using SplitApplyCombine: group
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @testset

@testset "belief_propagation (eltype=$elt)" for elt in (
        Float32, Float64, Complex{Float32}, Complex{Float64},
    )
    begin
        ITensors.disable_warn_order()
        g = named_grid((3, 3))
        s = siteinds("S=1/2", g)
        χ = 2
        rng = StableRNG(1234)
        ψ = random_tensornetwork(rng, elt, s; link_space = χ)
        ψψ = ψ ⊗ prime(dag(ψ); sites = [])
        pv = group(v -> first(v), vertices(ψψ))
        ptn = PartitionedGraph(ψψ, pv)
        # Build identity initial messages by hand: layer 1 is the ket
        # (original ψ) and layer 2 is the bra (`dag(prime(ψ; sites = []))`).
        # Pair the bra/ket link inds ordinally on each quotient edge so the
        # resulting `delta(b, dag(k))` has opposite QN flow.
        pairings = Dictionary{QuotientEdge, Pair{Vector{Index}, Vector{Index}}}()
        for pe in quotientedges(ptn)
            v_src, v_dst = parent(src(pe)), parent(dst(pe))
            for (v_from, v_to, e) in (
                    (v_src, v_dst, pe),
                    (v_dst, v_src, reverse(pe)),
                )
                bras = collect(commoninds(ψψ[(v_from, 2)], ψψ[(v_to, 2)]))
                kets = collect(commoninds(ψψ[(v_from, 1)], ψψ[(v_to, 1)]))
                set!(pairings, e, bras => kets)
            end
        end
        bpc = BeliefPropagationCache(ptn; messages = identity_messages(elt, pairings))

        #Test updating the tensors in the cache
        vket, vbra = ((1, 1), 1), ((1, 1), 2)
        A = bpc[vket]
        new_A = random_itensor(elt, inds(A))
        new_A_dag = ITensors.replaceind(
            dag(prime(new_A)), only(s[first(vket)])', only(s[first(vket)])
        )
        bpc[vket] = new_A
        bpc[vbra] = new_A_dag
        @test bpc[vket] == new_A
        @test bpc[vbra] == new_A_dag

        bpc = update(bpc; alg = "bp", maxiter = 25, tol = eps(real(elt)))
        #Test messages are converged
        for pe in quotientedges(bpc)
            @test message_diff(updated_message(bpc, pe), message(bpc, pe)) <
                10 * eps(real(elt))
            @test eltype(only(message(bpc, pe))) == elt
        end
        #Test updating the underlying tensornetwork in the cache
        v = first(vertices(ψψ))
        rng = StableRNG(1234)
        new_tensor = random_itensor(rng, inds(ψψ[v]))
        bpc_updated = update_factor(bpc, v, new_tensor)
        ψψ_updated = tensornetwork(bpc_updated)
        @test ψψ_updated[v] == new_tensor

        #Test forming a two-site RDM. Check it has the correct size, trace 1 and is PSD
        vs = [(2, 2), (2, 3)]

        # Prime the bra-ket shared site indices on the ket side at the
        # selected vertices, so the contracted RDM has open primed/unprimed
        # legs there. Mutates a copy of `ψψ` in place; no graph edits.
        ψψsplit = copy(ψψ)
        for v in vs
            common = commoninds(ψψsplit[(v, 1)], ψψsplit[(v, 2)])
            ψψsplit[(v, 2)] = prime(ψψsplit[(v, 2)], common)
        end
        env_tensors = environment(bpc, [(v, 2) for v in vs])
        rdm =
            contract(vcat(env_tensors, ITensor[ψψsplit[vp] for vp in [(v, 2) for v in vs]]))

        rdm = array(
            (rdm * combiner(inds(rdm; plev = 0)...)) * combiner(inds(rdm; plev = 1)...)
        )
        rdm /= tr(rdm)

        eigs = eigvals(rdm)
        @test size(rdm) == (2^length(vs), 2^length(vs))

        @test all(eig -> abs(imag(eig)) <= eps(real(elt)), eigs)
        @test all(eig -> real(eig) >= -eps(real(elt)), eigs)

        #Test edge case of network which evalutes to 0
        χ = 2
        g = named_grid((3, 1))
        rng = StableRNG(1234)
        ψ = random_tensornetwork(rng, elt, g; link_space = χ)
        ψ[(1, 1)] = 0 * ψ[(1, 1)]
        @test iszero(scalar(ψ; alg = "bp"))
    end
end

# Regression guard for `identity_messages`: a tiny QN-graded loopy
# network (4-cycle, S=1/2 sites) where the old single-leg `delta(i)`
# initialization collapsed messages to empty blocks and BP produced
# NaN. Two-leg `delta(b, k)` keeps the QN sectors aligned, so the
# QFN-routed `scalar` (which auto-uses `identity_messages` because QFN
# is structurally ψ-vs-ψ) must come back finite and nonzero.
@testset "QN-graded loopy BP — identity_messages regression" begin
    g = named_grid((2, 2); periodic = true)
    s = siteinds("S=1/2", g; conserve_qns = true)
    ψ = productstate(v -> isodd(sum(v)) ? "↑" : "↓", s)
    n2 = scalar(QuadraticFormNetwork(ψ); alg = "bp", cache_update_kwargs = (; maxiter = 25))
    @test isfinite(n2)
    @test !iszero(n2)
end
