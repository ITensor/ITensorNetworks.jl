# Density matrix algorithm and ttn_svd algorithm
"""
Approximate a `binary_tree_partition` into an output ITensorNetwork
with the same binary tree structure. `root` is the root vertex of the
pre-order depth-first-search traversal used to perform the truncations.
"""
function approx_itensornetwork(
  ::Algorithm"density_matrix",
  binary_tree_partition::DataGraph;
  root,
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  @assert is_tree(binary_tree_partition)
  @assert root in vertices(binary_tree_partition)
  @assert _is_rooted_directed_binary_tree(dfs_tree(binary_tree_partition, root))
  # The `binary_tree_partition` may contain multiple delta tensors to make sure
  # the partition has a binary tree structure. These delta tensors could hurt the
  # performance when computing density matrices so we remove them first.
  partition_wo_deltas = _contract_deltas_ignore_leaf_partitions(
    binary_tree_partition; root=root
  )
  return _approx_itensornetwork_density_matrix!(
    partition_wo_deltas,
    underlying_graph(binary_tree_partition);
    root,
    cutoff,
    maxdim,
    contraction_sequence_alg,
    contraction_sequence_kwargs,
  )
end

function approx_itensornetwork(
  ::Algorithm"ttn_svd",
  binary_tree_partition::DataGraph;
  root,
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  @assert is_tree(binary_tree_partition)
  @assert root in vertices(binary_tree_partition)
  @assert _is_rooted_directed_binary_tree(dfs_tree(binary_tree_partition, root))
  return _approx_itensornetwork_ttn_svd!(
    binary_tree_partition;
    root,
    cutoff,
    maxdim,
    contraction_sequence_alg,
    contraction_sequence_kwargs,
  )
end

"""
Approximate a given ITensorNetwork `tn` into an output ITensorNetwork
with a binary tree structure. The binary tree structure is defined based
on `inds_btree`, which is a directed binary tree DataGraph of indices.
"""
function approx_itensornetwork(
  alg::Union{Algorithm"density_matrix",Algorithm"ttn_svd"},
  tn::ITensorNetwork,
  inds_btree::DataGraph;
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg="optimal",
  contraction_sequence_kwargs=(;),
)
  par = _partition(tn, inds_btree; alg="mincut_recursive_bisection")
  output_tn, log_root_norm = approx_itensornetwork(
    alg,
    par;
    root=_root(inds_btree),
    cutoff=cutoff,
    maxdim=maxdim,
    contraction_sequence_alg=contraction_sequence_alg,
    contraction_sequence_kwargs=contraction_sequence_kwargs,
  )
  # Each leaf vertex in `output_tn` is adjacent to one output index.
  # We remove these leaf vertices so that each non-root vertex in `output_tn`
  # is an order 3 tensor.
  _rem_leaf_vertices!(
    output_tn;
    root=_root(inds_btree),
    contraction_sequence_alg=contraction_sequence_alg,
    contraction_sequence_kwargs=contraction_sequence_kwargs,
  )
  return output_tn, log_root_norm
end

"""
Approximate a given ITensorNetwork `tn` into an output ITensorNetwork with `output_structure`.
`output_structure` outputs a directed binary tree DataGraph defining the desired graph structure.
"""
function approx_itensornetwork(
  alg::Union{Algorithm"density_matrix",Algorithm"ttn_svd"},
  tn::ITensorNetwork,
  output_structure::Function=path_graph_structure;
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg="optimal",
  contraction_sequence_kwargs=(;),
)
  inds_btree = output_structure(tn)
  return approx_itensornetwork(
    alg,
    tn,
    inds_btree;
    cutoff=cutoff,
    maxdim=maxdim,
    contraction_sequence_alg=contraction_sequence_alg,
    contraction_sequence_kwargs=contraction_sequence_kwargs,
  )
end

# interface
function approx_itensornetwork(
  partitioned_tn::DataGraph;
  alg::String,
  root,
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg="optimal",
  contraction_sequence_kwargs=(;),
)
  return approx_itensornetwork(
    Algorithm(alg),
    partitioned_tn;
    root,
    cutoff,
    maxdim,
    contraction_sequence_alg,
    contraction_sequence_kwargs,
  )
end

function approx_itensornetwork(
  tn::ITensorNetwork,
  inds_btree::DataGraph;
  alg::String,
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg="optimal",
  contraction_sequence_kwargs=(;),
)
  return approx_itensornetwork(
    Algorithm(alg),
    tn,
    inds_btree;
    cutoff,
    maxdim,
    contraction_sequence_alg,
    contraction_sequence_kwargs,
  )
end

function approx_itensornetwork(
  tn::ITensorNetwork,
  output_structure::Function=path_graph_structure;
  alg::String,
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg="optimal",
  contraction_sequence_kwargs=(;),
)
  return approx_itensornetwork(
    Algorithm(alg),
    tn,
    output_structure;
    cutoff,
    maxdim,
    contraction_sequence_alg,
    contraction_sequence_kwargs,
  )
end
