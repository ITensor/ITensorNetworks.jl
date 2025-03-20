module ITensorNetworksTensorOperationsExt

using ITensors: ITensors, ITensor, dim, inds
using ITensorNetworks: ITensorNetworks
using NDTensors.AlgorithmSelection: @Algorithm_str
using TensorOperations: TensorOperations, optimaltree

function ITensorNetworks.contraction_sequence(::Algorithm"optimal", tn::Vector{ITensor})
  network = collect.(inds.(tn))
  inds_to_dims = Dict(i => dim(i) for i in unique(reduce(vcat, network)))
  seq, _ = optimaltree(network, inds_to_dims)
  return seq
end

end
