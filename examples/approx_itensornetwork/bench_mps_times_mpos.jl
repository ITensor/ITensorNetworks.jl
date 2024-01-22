using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using KaHyPar, Metis
using ITensorNetworks
using ITensorNetworks: ising_network, contract, _nested_vector_to_digraph
using OMEinsumContractionOrders
using AbstractTrees

include("utils.jl")

function mps_times_mpos(len_mps, num_mpo; link_space, physical_space)
  inds_net = IndsNetwork(named_grid((len_mps, num_mpo)))
  for j in 1:num_mpo
    for i in 1:(len_mps - 1)
      inds_net[(i, j) => (i + 1, j)] = [Index(link_space, "$(i)x$(j),$(i+1)x$(j)")]
    end
  end
  for i in 1:len_mps
    for j in 1:(num_mpo - 1)
      inds_net[(i, j) => (i, j + 1)] = [Index(physical_space, "$(i)x$(j),$(i)x$(j+1)")]
    end
  end
  inds_leaves = [Index(physical_space, "$(i)x$(num_mpo),out") for i in 1:len_mps]
  for i in 1:len_mps
    inds_net[(i, num_mpo)] = [inds_leaves[i]]
  end
  distribution = Uniform{Float64}(-1.0, 1.0)
  return randomITensorNetwork(distribution, inds_net), inds_leaves
end

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
# TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

len_mps = 30
num_mpo = 2
link_space = 80 # the largest seems 80, beyond which would get too slow.
maxdim = link_space
physical_space = 2
tree = "comb"

tn, inds_leaves = mps_times_mpos(
  len_mps, num_mpo; link_space=link_space, physical_space=physical_space
)
if tree == "mps"
  btree = _nested_vector_to_digraph(linear_sequence(inds_leaves))
else
  btree = _nested_vector_to_digraph(bipartite_sequence(inds_leaves))
end
# @info "tn", tn
# @info "inds_leaves is", inds_leaves
for alg in ["ttn_svd", "density_matrix"]
  @info "alg is", alg
  reset_timer!(ITensors.timer)
  out = @time bench(tn, btree; alg=alg, maxdim=maxdim)
  @info "out norm is", out[2]
  show(ITensors.timer)
  @info ""
end
