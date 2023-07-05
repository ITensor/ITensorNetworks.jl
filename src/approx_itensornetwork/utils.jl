"""
For a given ITensorNetwork `tn` and a `root` vertex, remove leaf vertices in the directed tree
with root `root` without changing the tensor represented by tn.
In particular, the tensor of each leaf vertex is contracted with the tensor of its parent vertex
to keep the tensor unchanged.
"""
function _rem_leaf_vertices!(
  tn::ITensorNetwork;
  root=first(vertices(tn)),
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  dfs_t = dfs_tree(tn, root)
  leaves = leaf_vertices(dfs_t)
  parents = [parent_vertex(dfs_t, leaf) for leaf in leaves]
  for (l, p) in zip(leaves, parents)
    tn[p] = _optcontract(
      [tn[p], tn[l]]; contraction_sequence_alg, contraction_sequence_kwargs
    )
    rem_vertex!(tn, l)
  end
end

"""
Contract of a vector of tensors, `network`, with a contraction sequence generated via sa_bipartite
# TODO: rewrite using `contract`
"""
function _optcontract(
  network::Vector; contraction_sequence_alg="optimal", contraction_sequence_kwargs=(;)
)
  @timeit_debug ITensors.timer "[approx_binary_tree_itensornetwork]: _optcontract" begin
    if length(network) == 0
      return ITensor(1.0)
    end
    @assert network isa Vector{ITensor}
    @timeit_debug ITensors.timer "[approx_binary_tree_itensornetwork]: contraction_sequence" begin
      seq = contraction_sequence(
        network; alg=contraction_sequence_alg, contraction_sequence_kwargs...
      )
    end
    output = contract(network; sequence=seq)
    return output
  end
end
