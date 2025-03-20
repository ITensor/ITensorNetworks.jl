module TensorOperationsExt
using ITensors: ITensors, ITensor, dim, inds
using TensorOperations: optimaltree

function ITensors.optimal_contraction_sequence(tensors::Vector{<:ITensor})
  network = collect.(inds.(tensors))
  inds_to_dims = Dict(i => dim(i) for i in unique(reduce(vcat, network)))
  seq, _ = optimaltree(network, inds_to_dims)
  return seq
end

end
