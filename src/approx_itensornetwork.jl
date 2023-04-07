"""
The struct contains cached density matrices and cached partial density matrices
for each edge / set of edges in the tensor network.

Density matrix example:
  Consider a tensor network below,
    1
    /\
   9  2
  /   /\
     3  6
    /|  /\
   4 5 7  8
  /  | |   \
  The density matrix for the edge `NamedEdge(2, 3)` squares the subgraph with vertices 3, 4, 5
     |
     3
    /|
   4 5
   | |
   4 5
   |/
   3
   |
  The density matrix for the edge `NamedEdge(5, 3)` squares the subgraph
    with vertices 1, 2, 3, 4, 6, 7, 8, 9
      1
      /\
     /  2
    /   /\
   /   3  6
  9   /|  /\
  |  4   7  8
  |  |   |  |
  |  4   7  8
  |  |/  | /
  |  3   6
  |  |  /
  |  | /
  |  2
  9 /
  |/
  1
  The density matrix for the edge `NamedEdge(4, 3)` squares the subgraph
    with vertices 1, 2, 3, 5, 6, 7, 8, 9
      1
      /\
     /  2
    /   /\
   /   3  6
  9   /|  /\
  |    5 7  8
  |    | |  |
  |    5 7  8
  |  |/  | /
  |  3   6
  |  |  /
  |  | /
  |  2
  9 /
  |/
  1

Partial density matrix example:
  Consider a tensor network below,
    1
    /\
   9  2
  /   /\
     3  6
    /|  /\
   4 5 7  8
  /  | |   \
  The partial density matrix for the Edge set `Set([NamedEdge(2, 3), NamedEdge(5, 3)])`
    squares the subgraph with vertices 4, and contract with the tensor 3
    |
    3
   /
  4 - 4 -
  The partial density matrix for the Edge set `Set([NamedEdge(4, 3), NamedEdge(5, 3)])`
    squares the subgraph with vertices 1, 2, 6, 7, 8, 9, and contract with the tensor 3
      1
      /\
     /  2
    /   /\
   /   3  6
  9   /|  /\
  |      7  8
  |      |  |
  |      7  8
  |      | /
  |      6
  |     /
  |  | /
  |  2
  9 /
  |/
  1
  The density matrix for the Edge set `Set([NamedEdge(4, 3), NamedEdge(2, 3)])`
    squares the subgraph with vertices 5. and contract with the tensor 3
    |
    3
   /
  5 - 5 -
"""
struct _DensityMatrixAlgCaches
  e_to_dm::Dict{NamedEdge,ITensor}
  es_to_pdm::Dict{Set{NamedEdge},ITensor}
end

function _DensityMatrixAlgCaches()
  e_to_dm = Dict{NamedEdge,ITensor}()
  es_to_pdm = Dict{Set{NamedEdge},ITensor}()
  return _DensityMatrixAlgCaches(e_to_dm, es_to_pdm)
end

"""
The struct stores data used in the density matrix algorithm.
  partition: The given tn partition
  out_tree: the binary tree structure of the output ITensorNetwork
  root: root vertex of the bfs_tree for truncation
  innerinds_to_sim: mapping each inner index of the tn represented by `partition` to a sim index
  caches: all the cached density matrices
"""
struct _DensityMartrixAlgGraph
  partition::DataGraph
  out_tree::NamedGraph
  root::Any
  innerinds_to_sim::Dict{<:Index,<:Index}
  caches::_DensityMatrixAlgCaches
end

function _DensityMartrixAlgGraph(partition::DataGraph, out_tree::NamedGraph, root::Any)
  innerinds = _commoninds(partition)
  sim_innerinds = [sim(ind) for ind in innerinds]
  return _DensityMartrixAlgGraph(
    partition,
    out_tree,
    root,
    Dict(zip(innerinds, sim_innerinds)),
    _DensityMatrixAlgCaches(),
  )
end

"""
Contract of a vector of tensors, `network`, with a contraction sequence generated via sa_bipartite
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

function _get_low_rank_projector(tensor, inds1, inds2; cutoff, maxdim)
  @assert length(inds(tensor)) <= 4
  @timeit_debug ITensors.timer "[approx_binary_tree_itensornetwork]: eigen" begin
    F = eigen(tensor, inds1, inds2; cutoff=cutoff, maxdim=maxdim, ishermitian=true)
  end
  return F.Vt
end

"""
Returns a dict that maps the partition's outinds that are adjacent to `partition[root]` to siminds
"""
function _densitymatrix_outinds_to_sim(partition, root)
  outinds = _noncommoninds(partition)
  outinds_root = intersect(outinds, noncommoninds(Vector{ITensor}(partition[root])...))
  outinds_root_to_sim = Dict(zip(outinds_root, [sim(ind) for ind in outinds_root]))
  return outinds_root_to_sim
end

"""
Replace the inds of partial_dm_tensor that are in keys of `inds_to_siminds` to the
corresponding value, and replace the inds that are in values of `inds_to_siminds`
to the corresponding key.
"""
function _sim(partial_dm_tensor::ITensor, inds_to_siminds)
  siminds_to_inds = Dict(zip(values(inds_to_siminds), keys(inds_to_siminds)))
  indices = keys(inds_to_siminds)
  indices = intersect(indices, inds(partial_dm_tensor))
  simindices = setdiff(inds(partial_dm_tensor), indices)
  reorder_inds = [indices..., simindices...]
  reorder_siminds = vcat(
    [inds_to_siminds[i] for i in indices], [siminds_to_inds[i] for i in simindices]
  )
  return replaceinds(partial_dm_tensor, reorder_inds => reorder_siminds)
end

"""
Update `caches.e_to_dm[e]` and `caches.es_to_pdm[es]`.
  caches: the caches of the density matrix algorithm.
  edge: the edge defining the density matrix
  children: the children vertices of `dst(edge)` in the dfs_tree
  network: the tensor network at vertex `dst(edge)`
  inds_to_sim: a dict mapping inds to sim inds
"""
function _update!(
  caches::_DensityMatrixAlgCaches,
  edge::NamedEdge,
  children::Vector,
  network::Vector{ITensor},
  inds_to_sim;
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  v = dst(edge)
  if haskey(caches.e_to_dm, edge)
    return nothing
  end
  child_to_dm = [c => caches.e_to_dm[NamedEdge(v, c)] for c in children]
  pdms = []
  for (child_v, dm_tensor) in child_to_dm
    es = [NamedEdge(src_v, v) for src_v in setdiff(children, child_v)]
    es = Set(vcat(es, [edge]))
    if !haskey(caches.es_to_pdm, es)
      caches.es_to_pdm[es] = _optcontract(
        [dm_tensor, network...]; contraction_sequence_alg, contraction_sequence_kwargs
      )
    end
    push!(pdms, caches.es_to_pdm[es])
  end
  if length(pdms) == 0
    sim_network = map(x -> replaceinds(x, inds_to_sim), network)
    sim_network = map(dag, sim_network)
    density_matrix = _optcontract(
      [network..., sim_network...]; contraction_sequence_alg, contraction_sequence_kwargs
    )
  elseif length(pdms) == 1
    sim_network = map(x -> replaceinds(x, inds_to_sim), network)
    sim_network = map(dag, sim_network)
    density_matrix = _optcontract(
      [pdms[1], sim_network...]; contraction_sequence_alg, contraction_sequence_kwargs
    )
  else
    simtensor = _sim(pdms[2], inds_to_sim)
    simtensor = dag(simtensor)
    density_matrix = _optcontract(
      [pdms[1], simtensor]; contraction_sequence_alg, contraction_sequence_kwargs
    )
  end
  caches.e_to_dm[edge] = density_matrix
  return nothing
end

"""
Perform truncation and remove `root` vertex in the `partition` and `out_tree`
of `alg_graph`.

Example:
  Consider an `alg_graph`` whose `out_tree` is shown below,
    1
    /\
   9  2
  /   /\
     3  6
    /|  /\
   4 5 7  8
  /  | |   \
  when `root = 4`, the output `out_tree` will be
    1
    /\
   9  2
  /   /\
     3  6
    /|  /\
     5 7  8
     | |   \
  and the returned tensor `U` will be the projector at vertex 4 in the output tn.
"""
function _rem_vertex!(
  alg_graph::_DensityMartrixAlgGraph,
  root;
  cutoff,
  maxdim,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  caches = alg_graph.caches
  outinds_root_to_sim = _densitymatrix_outinds_to_sim(alg_graph.partition, root)
  inds_to_sim = merge(alg_graph.innerinds_to_sim, outinds_root_to_sim)
  dm_dfs_tree = dfs_tree(alg_graph.out_tree, root)
  @assert length(child_vertices(dm_dfs_tree, root)) == 1
  for v in post_order_dfs_vertices(dm_dfs_tree, root)
    children = sort(child_vertices(dm_dfs_tree, v))
    @assert length(children) <= 2
    network = Vector{ITensor}(alg_graph.partition[v])
    _update!(
      caches,
      NamedEdge(parent_vertex(dm_dfs_tree, v), v),
      children,
      Vector{ITensor}(network),
      inds_to_sim;
      contraction_sequence_alg,
      contraction_sequence_kwargs,
    )
  end
  U = _get_low_rank_projector(
    caches.e_to_dm[NamedEdge(nothing, root)],
    collect(keys(outinds_root_to_sim)),
    collect(values(outinds_root_to_sim));
    cutoff,
    maxdim,
  )
  # update partition and out_tree
  root_tensor = _optcontract(
    [Vector{ITensor}(alg_graph.partition[root])..., dag(U)];
    contraction_sequence_alg,
    contraction_sequence_kwargs,
  )
  new_root = child_vertices(dm_dfs_tree, root)[1]
  alg_graph.partition[new_root] = disjoint_union(
    alg_graph.partition[new_root], ITensorNetwork([root_tensor])
  )
  rem_vertex!(alg_graph.partition, root)
  rem_vertex!(alg_graph.out_tree, root)
  # update es_to_pdm
  truncate_dfs_tree = dfs_tree(alg_graph.out_tree, alg_graph.root)
  for es in keys(caches.es_to_pdm)
    if dst(first(es)) == root
      delete!(caches.es_to_pdm, es)
    elseif dst(first(es)) == new_root
      parent_edge = NamedEdge(parent_vertex(truncate_dfs_tree, new_root), new_root)
      edge_to_remove = NamedEdge(root, new_root)
      if intersect(es, [parent_edge]) == []
        new_es = setdiff(es, [edge_to_remove])
        caches.es_to_pdm[new_es] = _optcontract(
          [caches.es_to_pdm[es], root_tensor];
          contraction_sequence_alg,
          contraction_sequence_kwargs,
        )
      end
      # Remove old caches since they won't be used anymore,
      # and removing them saves later contraction costs.
      delete!(caches.es_to_pdm, es)
    end
  end
  # update e_to_dm
  for edge in keys(caches.e_to_dm)
    if dst(edge) in [root, new_root]
      delete!(caches.e_to_dm, edge)
    end
  end
  return U
end

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
Approximate a `partition` into an output ITensorNetwork
with the binary tree structure defined by `out_tree`.
"""
function _approx_binary_tree_itensornetwork!(
  input_partition::DataGraph,
  out_tree::NamedGraph;
  root=first(vertices(partition)),
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  # Change type of each partition[v] since they will be updated
  # with potential data type chage.
  partition = DataGraph()
  for v in vertices(input_partition)
    add_vertex!(partition, v)
    partition[v] = ITensorNetwork{Any}(input_partition[v])
  end
  @assert sort(vertices(partition)) == sort(vertices(out_tree))
  alg_graph = _DensityMartrixAlgGraph(partition, out_tree, root)
  output_tn = ITensorNetwork()
  for v in post_order_dfs_vertices(out_tree, root)[1:(end - 1)]
    U = _rem_vertex!(
      alg_graph, v; cutoff, maxdim, contraction_sequence_alg, contraction_sequence_kwargs
    )
    add_vertex!(output_tn, v)
    output_tn[v] = U
  end
  @assert length(vertices(partition)) == 1
  add_vertex!(output_tn, root)
  root_tensor = _optcontract(
    Vector{ITensor}(partition[root]); contraction_sequence_alg, contraction_sequence_kwargs
  )
  root_norm = norm(root_tensor)
  root_tensor /= root_norm
  output_tn[root] = root_tensor
  return output_tn, log(root_norm)
end

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
  return _approx_binary_tree_itensornetwork!(
    partition_wo_deltas,
    underlying_graph(binary_tree_partition);
    root,
    cutoff,
    maxdim,
    contraction_sequence_alg,
    contraction_sequence_kwargs,
  )
end

"""
Approximate a given ITensorNetwork `tn` into an output ITensorNetwork with `output_structure`.
`output_structure` outputs a directed binary tree DataGraph defining the desired graph structure.
"""
function approx_itensornetwork(
  ::Algorithm"density_matrix",
  tn::ITensorNetwork,
  output_structure::Function=path_graph_structure;
  cutoff,
  maxdim,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  inds_btree = output_structure(tn)
  return approx_itensornetwork(
    tn,
    inds_btree;
    alg="density_matrix",
    cutoff=cutoff,
    maxdim=maxdim,
    contraction_sequence_alg=contraction_sequence_alg,
    contraction_sequence_kwargs=contraction_sequence_kwargs,
  )
end

"""
Approximate a given ITensorNetwork `tn` into an output ITensorNetwork
with a binary tree structure. The binary tree structure is defined based
on `inds_btree`, which is a directed binary tree DataGraph of indices.
"""
function approx_itensornetwork(
  ::Algorithm"density_matrix",
  tn::ITensorNetwork,
  inds_btree::DataGraph;
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg="optimal",
  contraction_sequence_kwargs=(;),
)
  par = partition(tn, inds_btree; alg="mincut_recursive_bisection")
  output_tn, log_root_norm = approx_itensornetwork(
    par;
    alg="density_matrix",
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
