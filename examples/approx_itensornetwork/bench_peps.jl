using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using KaHyPar, Metis
using ITensorNetworks
using ITensorNetworks: ising_network, contract, _nested_vector_to_digraph, _ansatz_tree
using OMEinsumContractionOrders
using AbstractTrees

include("utils.jl")

function peps(N; link_space)
  inds_net = IndsNetwork(named_grid(N); link_space=link_space)
  inds_leaves = []
  for i in 1:N[1]
    push!(inds_leaves, [Index(link_space, "$(i)x$(j),out") for j in 1:N[2]])
  end
  for i in 1:N[1]
    for j in 1:N[2]
      inds_net[(i, j)] = [inds_leaves[i][j]]
    end
  end
  distribution = Uniform{Float64}(-1.0, 1.0)
  return randomITensorNetwork(distribution, inds_net), inds_leaves
end

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
# TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

N = (10, 10) # (9, 9) is the largest for comb
link_space = 2
maxdim = 100
ansatz = "mps"

tn, inds_leaves = peps(N; link_space=link_space)
ortho_center = div(length(inds_leaves), 2, RoundDown)
btree = _ansatz_tree(inds_leaves, ansatz, ortho_center)
@info "tn", tn
@info "inds_leaves is", inds_leaves
for alg in ["density_matrix", "ttn_svd"]
  @info "alg is", alg
  reset_timer!(ITensors.timer)
  out = @time bench(tn, btree; alg=alg, maxdim=maxdim)
  @info "out norm is", out[2]
  show(ITensors.timer)
  @info ""
end
