"""
Approximate a `partition` into an output ITensorNetwork
with the binary tree structure defined by `out_tree` by
first transforming the partition into a TTN, then truncating
the ttn using a sequence of SVDs.
"""
function _approx_itensornetwork_ttn_svd!(
  input_partition::DataGraph;
  root=first(vertices(partition)),
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  tn = ITensorNetwork()
  for v in vertices(input_partition)
    add_vertex!(tn, v)
    tn[v] = _optcontract(
      Vector{ITensor}(input_partition[v]);
      contraction_sequence_alg=contraction_sequence_alg,
      contraction_sequence_kwargs=contraction_sequence_kwargs,
    )
  end
  truncate_ttn = truncate(TTN(tn); cutoff=cutoff, maxdim=maxdim, root_vertex=root)
  out_tn = ITensorNetwork(truncate_ttn)
  root_tensor = out_tn[root]
  root_norm = norm(root_tensor)
  root_tensor /= root_norm
  out_tn[root] = root_tensor
  return out_tn, log(root_norm)
end
