@eval module $(gensym())
using Compat: Compat
using Graphs: vertices
# Trigger package extension.
using ITensorNetworks:
  ITensorNetworks,
  BeliefPropagationCache,
  ⊗,
  @preserve_graph,
  combine_linkinds,
  contract,
  contraction_sequence,
  eachtensor,
  environment,
  inner_network,
  linkinds_combiners,
  message,
  partitioned_tensornetwork,
  random_tensornetwork,
  scalar,
  siteinds,
  split_index,
  tensornetwork,
  update,
  update_factor,
  updated_message,
  message_diff
using ITensors: ITensors, ITensor, combiner, dag, inds, inner, op, prime, random_itensor
using ITensorNetworks.ModelNetworks: ModelNetworks
using ITensors.NDTensors: array
using LinearAlgebra: eigvals, tr
using NamedGraphs: NamedEdge, NamedGraph
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedges
using SplitApplyCombine: group
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @testset

@testset "belief_propagation (eltype=$elt)" for elt in (
  Float32, Float64, Complex{Float32}, Complex{Float64}
)
  begin
    ITensors.disable_warn_order()
    g = named_grid((3, 3))
    s = siteinds("S=1/2", g)
    χ = 2
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, elt, s; link_space=χ)
    ψψ = ψ ⊗ prime(dag(ψ); sites=[])
    bpc = BeliefPropagationCache(ψψ, group(v -> first(v), vertices(ψψ)))

    #Test updating the tensors in the cache
    vket, vbra = ((1, 1), 1), ((1, 1), 2)
    A = bpc[vket]
    new_A = random_itensor(elt, inds(A))
    new_A_dag = ITensors.replaceind(
      dag(prime(new_A)), only(s[first(vket)])', only(s[first(vket)])
    )
    @preserve_graph bpc[vket] = new_A
    @preserve_graph bpc[vbra] = new_A_dag
    @test bpc[vket] == new_A
    @test bpc[vbra] == new_A_dag

    bpc = update(bpc; maxiter=25, tol=eps(real(elt)))
    #Test messages are converged
    for pe in partitionedges(bpc)
      @test message_diff(updated_message(bpc, pe), message(bpc, pe)) < 10 * eps(real(elt))
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

    ψψsplit = split_index(ψψ, NamedEdge.([(v, 1) => (v, 2) for v in vs]))
    env_tensors = environment(bpc, [(v, 2) for v in vs])
    rdm = contract(vcat(env_tensors, ITensor[ψψsplit[vp] for vp in [(v, 2) for v in vs]]))

    rdm = array((rdm * combiner(inds(rdm; plev=0)...)) * combiner(inds(rdm; plev=1)...))
    rdm /= tr(rdm)

    eigs = eigvals(rdm)
    @test size(rdm) == (2^length(vs), 2^length(vs))

    @test all(eig -> abs(imag(eig)) <= eps(real(elt)), eigs)
    @test all(eig -> real(eig) >= -eps(real(elt)), eigs)

    #Test edge case of network which evalutes to 0
    χ = 2
    g = named_grid((3, 1))
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, elt, g; link_space=χ)
    ψ[(1, 1)] = 0 * ψ[(1, 1)]
    @test iszero(scalar(ψ; alg="bp"))
  end
end
end
