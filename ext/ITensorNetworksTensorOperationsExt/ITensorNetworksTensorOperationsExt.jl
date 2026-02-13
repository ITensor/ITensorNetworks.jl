module ITensorNetworksTensorOperationsExt

using ITensorNetworks: ITensorNetworks, ITensorList
using ITensors: ITensors, ITensor, dim, inds
using NDTensors.AlgorithmSelection: @Algorithm_str
using TensorOperations: TensorOperations, optimaltree

function ITensorNetworks.contraction_sequence(::Algorithm"optimal", tn::ITensorList)
    network = collect.(inds.(tn))
    #Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(i => Float64(dim(i)) for i in unique(reduce(vcat, network)))
    seq, _ = optimaltree(network, inds_to_dims)
    return seq
end

end
