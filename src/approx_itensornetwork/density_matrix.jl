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

function _get_low_rank_projector(tensor, inds1, inds2; cutoff, maxdim)
  @assert length(inds(tensor)) <= 4
  # t00 = time()
  @timeit_debug ITensors.timer "[approx_binary_tree_itensornetwork]: eigen" begin
    F = eigen(tensor, inds1, inds2; cutoff=cutoff, maxdim=maxdim, ishermitian=true)
  end
  # t11 = time() - t00
  # @info "size of U", size(F.V), "size of diag", size(F.D), "costs", t11
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
  density_matrix_alg="qr_svd",
)
  caches = alg_graph.caches
  dm_dfs_tree = dfs_tree(alg_graph.out_tree, root)
  U = _update_cache_w_low_rank_projector!(
    Algorithm(density_matrix_alg),
    alg_graph,
    root;
    cutoff,
    maxdim,
    contraction_sequence_alg,
    contraction_sequence_kwargs,
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
  # Remove all partial density matrices that contain `root`,
  # since `root` has been removed from the graph.
  for es in filter(es -> dst(first(es)) == root, keys(caches.es_to_pdm))
    delete!(caches.es_to_pdm, es)
  end
  for es in filter(es -> dst(first(es)) == new_root, keys(caches.es_to_pdm))
    parent_edge = NamedEdge(parent_vertex(truncate_dfs_tree, new_root), new_root)
    edge_to_remove = NamedEdge(root, new_root)
    # The pdm represented by `new_es` will be used to
    # update the sibling vertex of `root`.
    if intersect(es, Set([parent_edge])) == Set()
      new_es = setdiff(es, [edge_to_remove])
      if new_es == Set()
        new_es = Set([NamedEdge(nothing, new_root)])
      end
      @assert length(new_es) >= 1
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
  # update e_to_dm
  for edge in filter(e -> dst(e) in [root, new_root], keys(caches.e_to_dm))
    delete!(caches.e_to_dm, edge)
  end
  return U
end

function _update_cache_w_low_rank_projector!(
  ::Algorithm"direct_eigen",
  alg_graph::_DensityMartrixAlgGraph,
  root;
  cutoff,
  maxdim,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  dm_dfs_tree = dfs_tree(alg_graph.out_tree, root)
  outinds_root_to_sim = _densitymatrix_outinds_to_sim(alg_graph.partition, root)
  # For keys that appear in both dicts, the value in
  # `outinds_root_to_sim` is used.
  inds_to_sim = merge(alg_graph.innerinds_to_sim, outinds_root_to_sim)
  @assert length(child_vertices(dm_dfs_tree, root)) == 1
  for v in post_order_dfs_vertices(dm_dfs_tree, root)
    children = sort(child_vertices(dm_dfs_tree, v))
    @assert length(children) <= 2
    network = Vector{ITensor}(alg_graph.partition[v])
    _update!(
      alg_graph.caches,
      NamedEdge(parent_vertex(dm_dfs_tree, v), v),
      children,
      Vector{ITensor}(network),
      inds_to_sim;
      contraction_sequence_alg,
      contraction_sequence_kwargs,
    )
  end
  return U = _get_low_rank_projector(
    alg_graph.caches.e_to_dm[NamedEdge(nothing, root)],
    collect(keys(outinds_root_to_sim)),
    collect(values(outinds_root_to_sim));
    cutoff,
    maxdim,
  )
end

function _update_cache_w_low_rank_projector!(
  ::Algorithm"qr_svd",
  alg_graph::_DensityMartrixAlgGraph,
  root;
  cutoff,
  maxdim,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  dm_dfs_tree = dfs_tree(alg_graph.out_tree, root)
  outinds_root_to_sim = _densitymatrix_outinds_to_sim(alg_graph.partition, root)
  # For keys that appear in both dicts, the value in
  # `outinds_root_to_sim` is used.
  inds_to_sim = merge(alg_graph.innerinds_to_sim, outinds_root_to_sim)
  @assert length(child_vertices(dm_dfs_tree, root)) == 1
  # Note: here we are not updating the density matrix of `root`
  traversal = post_order_dfs_vertices(dm_dfs_tree, root)[1:(end - 1)]
  for v in traversal
    children = sort(child_vertices(dm_dfs_tree, v))
    @assert length(children) <= 2
    network = Vector{ITensor}(alg_graph.partition[v])
    _update!(
      alg_graph.caches,
      NamedEdge(parent_vertex(dm_dfs_tree, v), v),
      children,
      Vector{ITensor}(network),
      inds_to_sim;
      contraction_sequence_alg,
      contraction_sequence_kwargs,
    )
  end
  root_t = _optcontract(
    Vector{ITensor}(alg_graph.partition[root]);
    contraction_sequence_alg,
    contraction_sequence_kwargs,
  )
  Q, R = factorize(
    root_t, collect(keys(outinds_root_to_sim))...; which_decomp="qr", ortho="left"
  )
  qr_commonind = commoninds(Q, R)[1]
  R_sim = replaceinds(R, inds_to_sim)
  R_sim = replaceinds(R_sim, qr_commonind => sim(qr_commonind))
  dm = alg_graph.caches.e_to_dm[NamedEdge(
    parent_vertex(dm_dfs_tree, traversal[end]), traversal[end]
  )]
  R = _optcontract([R, dm, R_sim]; contraction_sequence_alg, contraction_sequence_kwargs)
  U2 = _get_low_rank_projector(
    R, [qr_commonind], setdiff(inds(R), [qr_commonind]); cutoff, maxdim
  )
  # U2, _, _ = svd(R, commoninds(Q, R)...; maxdim=maxdim, cutoff=cutoff)
  return _optcontract([Q, U2]; contraction_sequence_alg, contraction_sequence_kwargs)
end

"""
Approximate a `partition` into an output ITensorNetwork
with the binary tree structure defined by `out_tree`.
"""
function _approx_itensornetwork_density_matrix!(
  input_partition::DataGraph,
  out_tree::NamedGraph;
  root=first(vertices(partition)),
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
  density_matrix_alg="direct_eigen",
)
  @assert density_matrix_alg in ["direct_eigen", "qr_svd"]
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
      alg_graph,
      v;
      cutoff,
      maxdim,
      contraction_sequence_alg,
      contraction_sequence_kwargs,
      density_matrix_alg,
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
