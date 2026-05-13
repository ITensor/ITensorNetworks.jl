using Compat: Compat
using Graphs: vertices
using ITensorNetworks: ITensorNetworks, BeliefPropagationCache, QuadraticFormNetwork,
    bra_vertex, contract, contraction_sequence, default_partitioned_vertices, environment,
    identity_messages, ket_vertex, message, message_diff, norm_sqr_network, operator_vertex,
    partitioned_tensornetwork, scalar, siteinds, tensornetwork, update, update_factor,
    updated_message
include("utils.jl")
using ITensors.NDTensors: array
using ITensors: ITensors, ITensor, combiner, dag, inds, inner, prime, random_itensor
using LinearAlgebra: eigvals, tr
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionedGraph, quotientedges
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
        ψψ = norm_sqr_network(ψ)
        ptn = PartitionedGraph(ψψ, default_partitioned_vertices(ψψ))
        bpc = BeliefPropagationCache(ptn; messages = identity_messages(ψψ, ptn))

        # Test updating the tensors in the cache. QFN bra has both site
        # and link inds primed (relative to the ket), so the bra-side
        # tensor is constructed as `dag(prime(new_A))` directly.
        v = (1, 1)
        vket = ket_vertex(ψψ, v)
        vbra = bra_vertex(ψψ, v)
        A = bpc[vket]
        new_A = random_itensor(elt, inds(A))
        new_A_dag = dag(prime(new_A))
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
        v_any = first(vertices(ψψ))
        rng = StableRNG(1234)
        new_tensor = random_itensor(rng, inds(ψψ[v_any]))
        bpc_updated = update_factor(bpc, v_any, new_tensor)
        ψψ_updated = tensornetwork(bpc_updated)
        @test ψψ_updated[v_any] == new_tensor

        # Two-site RDM at `vs`: ask for the environment around all three
        # layers (bra/ket/operator) at `vs` so the BP env is just incoming
        # messages, then contract with only the bra and ket tensors at
        # `vs` — dropping the per-site identity operator at those
        # vertices leaves the bra (primed) and ket (unprimed) site inds
        # open, which is exactly the RDM.
        vs = [(2, 2), (2, 3)]
        env_vs = vcat(
            [bra_vertex(ψψ, v) for v in vs],
            [ket_vertex(ψψ, v) for v in vs],
            [operator_vertex(ψψ, v) for v in vs]
        )
        env_tensors = environment(bpc, env_vs)
        local_tensors = vcat(
            ITensor[ψψ[bra_vertex(ψψ, v)] for v in vs],
            ITensor[ψψ[ket_vertex(ψψ, v)] for v in vs]
        )
        rdm = contract(vcat(env_tensors, local_tensors))
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
