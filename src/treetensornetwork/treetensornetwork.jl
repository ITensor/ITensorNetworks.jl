"""
    TreeTensorNetwork <: AbstractITensorNetwork

# Fields

- itensor_network::ITensorNetwork
- ortho_lims::Vector{Tuple}: A vector of vertices defining the orthogonality limits.

"""
struct TreeTensorNetwork <: AbstractITensorNetwork
  itensor_network::ITensorNetwork
  ortho_lims::Vector{Tuple}
end

const TTN = TreeTensorNetwork


