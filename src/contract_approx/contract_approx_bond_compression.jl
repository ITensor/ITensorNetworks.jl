
# contract(tn::Vector{ITensor}, sequence::Int) = tn[sequence]
# function contract(tn::Vector{ITensor}, sequence::AbstractVector; kwargs...)::ITensor
#   return contract(contract.((tn,), sequence)...; kwargs...)
# end

# Bond compression algorithm from https://arxiv.org/abs/2206.07044.
# A good sequence is a `post_order_dfs_edges` of a `bfs_tree` of the network for grid networks,
# using the vertex with the largest centrality as the source vertex of the spanning tree.
function contract_approx(::Algorithm"bond_compression", tn::AbstractITensorNetwork; sequence, maxdim, cutoff)
  # Recurse through the contraction sequence
  # Make an edge list from a contraction sequence!
end
