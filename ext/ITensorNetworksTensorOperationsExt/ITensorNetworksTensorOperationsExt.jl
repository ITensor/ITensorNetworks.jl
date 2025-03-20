module ITensorNetworksTensorOperationsExt

using ITensors: ITensors, ITensor, dim, inds
using ITensorNetworks: ITensorNetworks
using TensorOperations: optimaltree
using NDTensors.AlgorithmSelection: @Algorithm_str

function ITensorNetworks.contraction_sequence(::Algorithm"optimal", tn::Vector{ITensor})
  return optimal_contraction_sequence(tn)
end

function optimal_contraction_sequence(tensors::Vector{<:ITensor})
  network = collect.(inds.(tensors))
  inds_to_dims = Dict(i => dim(i) for i in unique(reduce(vcat, network)))
  seq, _ = optimaltree(network, inds_to_dims)
  return seq
end

end
