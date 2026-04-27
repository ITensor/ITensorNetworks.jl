# Developer Methods

## Form Networks

#### AbstractFormNetwork

* Required-to-implement abstract interface — each concrete `AbstractFormNetwork` subtype
  defines these (`formnetworks/abstractformnetwork.jl`):
  ```julia
  dual_index_map(f::AbstractFormNetwork)
  tensornetwork(f::AbstractFormNetwork)
  copy(f::AbstractFormNetwork)
  operator_vertex_suffix(f::AbstractFormNetwork)
  bra_vertex_suffix(f::AbstractFormNetwork)
  ket_vertex_suffix(f::AbstractFormNetwork)
  ```

* Graph plumbing forwarded to the underlying tensor network (`formnetworks/abstractformnetwork.jl`):
  ```julia
  data_graph(f::AbstractFormNetwork)
  data_graph_type(f::AbstractFormNetwork)
  ```

* Lists of vertices in each role: those tagged with the operator/bra/ket suffix (`formnetworks/abstractformnetwork.jl`):
  ```julia
  operator_vertices(f::AbstractFormNetwork)
  bra_vertices(f::AbstractFormNetwork)
  ket_vertices(f::AbstractFormNetwork)
  ```

* Vertex-renaming functions: closures `v -> (v, suffix)` that map an original-state
  vertex to its operator/bra/ket-tagged counterpart (`formnetworks/abstractformnetwork.jl`):
  ```julia
  operator_vertex_map(f::AbstractFormNetwork)
  bra_vertex_map(f::AbstractFormNetwork)
  ket_vertex_map(f::AbstractFormNetwork)
  ```

* Apply the corresponding vertex map to a single vertex `v` (`formnetworks/abstractformnetwork.jl`):
  ```julia
  operator_vertex(f::AbstractFormNetwork, v)
  bra_vertex(f::AbstractFormNetwork, v)
  ket_vertex(f::AbstractFormNetwork, v)
  ```

#### LinearFormNetwork

* Construct a `LinearFormNetwork` representing `⟨bra|ket⟩`. Optional suffix kwargs and
  a `dual_link_index_map` (default `prime`) control how bra link indices are made
  distinct from ket link indices (`formnetworks/linearformnetwork.jl`):
  ```julia
  LinearFormNetwork(bra::AbstractITensorNetwork, ket::AbstractITensorNetwork; bra_vertex_suffix, ket_vertex_suffix, dual_link_index_map)
  ```

* Construct a `LinearFormNetwork` from an existing `BilinearFormNetwork` by absorbing
  the operator into the bra side (`formnetworks/linearformnetwork.jl`):
  ```julia
  LinearFormNetwork(blf::BilinearFormNetwork)
  ```

* Suffix and tensor-network accessors (`formnetworks/linearformnetwork.jl`):
  ```julia
  bra_vertex_suffix(lf::LinearFormNetwork)
  ket_vertex_suffix(lf::LinearFormNetwork)
  tensornetwork(lf::LinearFormNetwork)
  ```

* Copy a `LinearFormNetwork` (deep-copies the underlying tensor network) (`formnetworks/linearformnetwork.jl`):
  ```julia
  copy(lf::LinearFormNetwork)
  ```

* Replace the ket-side tensor at the original vertex `original_ket_state_vertex` with
  `ket_state` (graph-preserving update) (`formnetworks/linearformnetwork.jl`):
  ```julia
  update(lf::LinearFormNetwork, original_ket_state_vertex, ket_state::ITensor)
  ```

#### BilinearFormNetwork

* Construct a `BilinearFormNetwork` representing `⟨bra|operator|ket⟩`. Optional suffix
  kwargs and `dual_site_index_map` (default `prime`) / `dual_link_index_map` (default
  `sim`) control how bra indices are distinguished from ket indices (`formnetworks/bilinearformnetwork.jl`):
  ```julia
  BilinearFormNetwork(operator::AbstractITensorNetwork, bra::AbstractITensorNetwork, ket::AbstractITensorNetwork; operator_vertex_suffix, bra_vertex_suffix, ket_vertex_suffix, dual_site_index_map, dual_link_index_map)
  ```

* Construct from `bra` and `ket` only — the operator network is built automatically
  as a per-vertex identity from `siteinds(ket)` to `dual_site_index_map(siteinds(ket))` (`formnetworks/bilinearformnetwork.jl`):
  ```julia
  BilinearFormNetwork(bra::AbstractITensorNetwork, ket::AbstractITensorNetwork; dual_site_index_map, kwargs...)
  ```

* Build the identity ITensor mapping `first(i_pair) → last(i_pair)` for each pair, used
  to assemble the auto-generated operator network (`formnetworks/bilinearformnetwork.jl`):
  ```julia
  itensor_identity_map(elt::Type, i_pairs::Vector)
  itensor_identity_map(i_pairs::Vector)
  ```

* Suffix and tensor-network accessors (`formnetworks/bilinearformnetwork.jl`):
  ```julia
  operator_vertex_suffix(blf::BilinearFormNetwork)
  bra_vertex_suffix(blf::BilinearFormNetwork)
  ket_vertex_suffix(blf::BilinearFormNetwork)
  tensornetwork(blf::BilinearFormNetwork)
  ```

* Copy a `BilinearFormNetwork` (`formnetworks/bilinearformnetwork.jl`):
  ```julia
  copy(blf::BilinearFormNetwork)
  ```

* Replace the bra and ket tensors at the original vertices with `bra_state` and
  `ket_state` (graph-preserving update) (`formnetworks/bilinearformnetwork.jl`):
  ```julia
  update(blf::BilinearFormNetwork, original_bra_state_vertex, original_ket_state_vertex, bra_state::ITensor, ket_state::ITensor)
  ```

#### QuadraticFormNetwork

* Construct a `QuadraticFormNetwork` representing `⟨ψ|operator|ψ⟩` (or `⟨ψ|ψ⟩` if the
  operator is omitted). Internally wraps a `BilinearFormNetwork` whose bra is the dual
  of `ket`. `dual_index_map` (default `prime`) / `dual_inv_index_map` (default `noprime`)
  control how bra indices are produced from ket indices (`formnetworks/quadraticformnetwork.jl`):
  ```julia
  QuadraticFormNetwork(operator::AbstractITensorNetwork, ket::AbstractITensorNetwork; dual_index_map, dual_inv_index_map, kwargs...)
  QuadraticFormNetwork(ket::AbstractITensorNetwork; dual_index_map, dual_inv_index_map, kwargs...)
  ```

* Access the underlying `BilinearFormNetwork` and the index-map functions (`formnetworks/quadraticformnetwork.jl`):
  ```julia
  bilinear_formnetwork(qf::QuadraticFormNetwork)
  dual_index_map(qf::QuadraticFormNetwork)
  dual_inv_index_map(qf::QuadraticFormNetwork)
  ```

* Forwarded accessors — these queries are answered by the inner `BilinearFormNetwork` (`formnetworks/quadraticformnetwork.jl`):
  ```julia
  operator_vertex_suffix(qf::QuadraticFormNetwork)
  bra_vertex_suffix(qf::QuadraticFormNetwork)
  ket_vertex_suffix(qf::QuadraticFormNetwork)
  tensornetwork(qf::QuadraticFormNetwork)
  data_graph(qf::QuadraticFormNetwork)
  data_graph_type(qf::QuadraticFormNetwork)
  ```

* Copy a `QuadraticFormNetwork` (deep-copies the inner bilinear form) (`formnetworks/quadraticformnetwork.jl`):
  ```julia
  copy(qf::QuadraticFormNetwork)
  ```

* Replace the tensor at the original vertex with `ket_state` — the bra-side tensor is
  generated automatically by applying `dual_index_map` to the dag of `ket_state` (`formnetworks/quadraticformnetwork.jl`):
  ```julia
  update(qf::QuadraticFormNetwork, original_state_vertex, ket_state::ITensor)
  ```

## Caches for BP

#### Edge Sequence Helpers

* Build an edge traversal sequence over `g`, dispatched by `alg` (default `"forest_cover"`).
  Directed graphs are handled by undirecting first (`edge_sequences.jl`):
  ```julia
  edge_sequence(g; alg, kwargs...)
  edge_sequence(alg::Algorithm, g; kwargs...)
  ```

* Tree-traversal sequence: cover `g` with a forest, then for each tree push a post-order
  DFS sweep followed by its reverse — gives a back-and-forth sequence covering every edge (`edge_sequences.jl`):
  ```julia
  edge_sequence(::Algorithm"forest_cover", g; root_vertex = GraphsExtensions.default_root_vertex)
  ```

* Parallel sequence: each edge (and its reverse) as its own one-edge group, suitable
  for parallel BP updates (`edge_sequences.jl`):
  ```julia
  edge_sequence(::Algorithm"parallel", g)
  ```

#### AbstractBeliefPropagationCache

* Required-to-implement abstract interface — concrete subtypes must define these
  (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  # How many of these are user-facing versus internal?
  setindex!(bpc::AbstractBeliefPropagationCache, factor::ITensor, vertex)
  partitioned_tensornetwork(bpc::AbstractBeliefPropagationCache)
  messages(bpc::AbstractBeliefPropagationCache)
  copy(bpc::AbstractBeliefPropagationCache)
  partitions(bpc::AbstractBeliefPropagationCache)
  quotientedges(bpc::AbstractBeliefPropagationCache)
  partitioned_vertices(bpc::AbstractBeliefPropagationCache)
  environment(bpc::AbstractBeliefPropagationCache, verts::Vector; kwargs...)
  region_scalar(bpc::AbstractBeliefPropagationCache, pv::QuotientVertex; kwargs...)
  region_scalar(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge; kwargs...)
  ```

* Forward type plumbing to the underlying tensor network and access the unpartitioned
  network behind a cache (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  similar_type(bpc::AbstractBeliefPropagationCache)
  data_graph_type(bpc::AbstractBeliefPropagationCache)
  data_graph(bpc::AbstractBeliefPropagationCache)
  tensornetwork(bpc::AbstractBeliefPropagationCache)
  scalartype(bpc::AbstractBeliefPropagationCache)
  ```

* Partition-graph queries forwarded to the underlying `PartitionedGraph` (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  vertices(bpc::AbstractBeliefPropagationCache)
  quotient_graph(bpc::AbstractBeliefPropagationCache)
  quotientedge(bpc::AbstractBeliefPropagationCache, edge::AbstractEdge)
  quotientvertices(bpc::AbstractBeliefPropagationCache)
  quotientvertices(bpc::AbstractBeliefPropagationCache, vs)
  boundary_quotientedges(bpc::AbstractBeliefPropagationCache, quotientvertices; kwargs...)
  boundary_quotientedges(bpc::AbstractBeliefPropagationCache, quotientvertex::QuotientVertex; kwargs...)
  linkinds(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge)
  ```

* Vertex tensors (factors) at given vertices, or for all vertices in the given
  partitions (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  factors(bpc::AbstractBeliefPropagationCache, verts::Vector)
  factors(bpc::AbstractBeliefPropagationCache, partition_verts::Vector{<:QuotientVertex})
  factors(bpc::AbstractBeliefPropagationCache, partition_vertex::QuotientVertex)
  ```

* Out-of-place factor updates — replace one or many vertex tensors, returning a new cache (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  update_factor(bpc, vertex, factor)
  update_factors(bpc::AbstractBeliefPropagationCache, factors)
  ```

* Apply a function to every (or a chosen subset of) factor tensors (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  map_factors(f, bpc::AbstractBeliefPropagationCache, vs = vertices(bpc))
  ```

* Read messages from the cache: a single message (with default fallback when missing),
  or a vector of messages on the given edges (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  message(bpc::AbstractBeliefPropagationCache, edge::QuotientEdge; kwargs...)
  messages(bpc::AbstractBeliefPropagationCache, edges; kwargs...)
  ```

* Set messages — in-place (`!`) variants mutate the cache, the non-`!` variants return
  a new cache (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  set_message!(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge, message)
  set_messages!(bpc::AbstractBeliefPropagationCache, quotientedges_messages)
  set_message(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge, message)
  set_messages(bpc::AbstractBeliefPropagationCache, quotientedges_messages)
  ```

* Delete messages on chosen edges (or all edges); same in-place vs out-of-place
  convention as `set_message[s]` (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  delete_message!(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge)
  delete_messages!(bpc::AbstractBeliefPropagationCache, pes::Vector{<:QuotientEdge} = keys(messages(bpc)))
  delete_message(bpc::AbstractBeliefPropagationCache, pe::QuotientEdge)
  delete_messages(bpc::AbstractBeliefPropagationCache, pes::Vector{<:QuotientEdge} = keys(messages(bpc)))
  ```

* Apply a function to every (or a chosen subset of) message tensors (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  map_messages(f, bpc::AbstractBeliefPropagationCache, pes = collect(keys(messages(bpc))))
  ```

* Collect the messages flowing into a partition vertex (or set of partition vertices),
  optionally ignoring some edges (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  incoming_messages(bpc::AbstractBeliefPropagationCache, partition_vertices::Vector{<:QuotientVertex}; ignore_edges = ())
  incoming_messages(bpc::AbstractBeliefPropagationCache, partition_vertex::QuotientVertex; kwargs...)
  ```

* Convergence proxy: `1 - |⟨a|b⟩|²` between two normalized contracted messages (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  message_diff(message_a::Vector{ITensor}, message_b::Vector{ITensor})
  ```

* Adapt support — propagate `adapt(to, ·)` over messages, factors, or both (used for
  GPU/eltype migration) (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  adapt_messages(to, bpc::AbstractBeliefPropagationCache, args...)
  adapt_factors(to, bpc::AbstractBeliefPropagationCache, args...)
  adapt_structure(to, bpc::AbstractBeliefPropagationCache)
  ```

* Region scalars per partition vertex / partition edge, and the convenient pair of
  vectors used by `logscalar` (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  vertex_scalars(bpc::AbstractBeliefPropagationCache, pvs = partitions(bpc); kwargs...)
  edge_scalars(bpc::AbstractBeliefPropagationCache, pes = quotientedges(bpc); kwargs...)
  scalar_factors_quotient(bpc::AbstractBeliefPropagationCache)
  ```

* The (log of the) BP estimate of the network scalar `⟨tn⟩` from current messages (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  logscalar(bpc::AbstractBeliefPropagationCache)
  scalar(bpc::AbstractBeliefPropagationCache)
  ```

* Compute an updated message on `edge` — either via straight contraction (with optional
  normalization), via an `adapt`-then-update wrapper, or via the user-facing dispatcher
  that fills in defaults (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  updated_message(alg::Algorithm"contract", bpc::AbstractBeliefPropagationCache, edge::QuotientEdge)
  updated_message(alg::Algorithm"adapt_update", bpc::AbstractBeliefPropagationCache, edge::QuotientEdge)
  updated_message(bpc::AbstractBeliefPropagationCache, edge::QuotientEdge; alg, kwargs...)
  ```

* Compute the updated message and write it back into a copied cache (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  update_message(message_update_alg::Algorithm, bpc::AbstractBeliefPropagationCache, edge::QuotientEdge)
  ```

* One BP iteration — sequential over a list of edges, or in parallel groups of edges,
  with an optional accumulator for per-iteration message diffs (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  update_iteration(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache, edges::Vector; update_diff!)
  update_iteration(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache, edge_groups::Vector{<:Vector{<:QuotientEdge}}; update_diff!)
  ```

* Drive BP to convergence — either the algorithm-specialized core or the user-facing
  dispatcher that fills in `maxiter`, `tol`, and the edge sequence (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  update(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache)
  update(bpc::AbstractBeliefPropagationCache; alg, kwargs...)
  ```

* Rescale messages on a single bond, on the given bonds, or on every bond, so that
  `region_scalar` of the bond is 1 (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  rescale_messages(bp_cache::AbstractBeliefPropagationCache, quotientedge::QuotientEdge)
  rescale_messages(bp_cache::AbstractBeliefPropagationCache, pes)
  rescale_messages(bp_cache::AbstractBeliefPropagationCache)
  ```

* Rescale the vertex tensors inside one partition, a list of partitions, or every
  partition, so that the partition's region scalar is 1 (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  rescale_partition(bpc::AbstractBeliefPropagationCache, partition, args...; kwargs...)
  rescale_partitions(bpc::AbstractBeliefPropagationCache, partitions::Vector; verts)
  rescale_partitions(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
  ```

* Convenience — rescale messages and then partitions in one call (`caches/abstractbeliefpropagationcache.jl`):
  ```julia
  rescale(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
  ```

#### BeliefPropagationCache

* Construct a `BeliefPropagationCache` from an
  `AbstractITensorNetwork` plus a partitioning, or just a network (using a default
  partition) (`caches/beliefpropagationcache.jl`):
  ```julia
  BeliefPropagationCache(tn::AbstractITensorNetwork, partitioned_vertices; kwargs...)
  BeliefPropagationCache(tn::AbstractITensorNetwork; partitioned_vertices, kwargs...)
  # Deprecate this version?
  BeliefPropagationCache(ptn::PartitionedGraph; messages = default_messages(ptn))
  ```

* Given a tensor network, returns a `BeliefPropagationCache` (`caches/beliefpropagationcache.jl`):
  ```julia
  cache(alg::Algorithm"bp", tn; kwargs...)
  ```

* Copy cache and access messages (`caches/beliefpropagationcache.jl`):
  ```julia
  copy(bp_cache::BeliefPropagationCache)
  messages(bp_cache::BeliefPropagationCache)
  setindex!(bpc::BeliefPropagationCache, factor::ITensor, vertex)
  ```

* Partition-graph related queries — list partitions, quotient edges between partitions, and
  the vertex-to-partition mapping (`caches/beliefpropagationcache.jl`):
  ```julia
  partitioned_tensornetwork(bp_cache::BeliefPropagationCache)
  partitions(bpc::BeliefPropagationCache)
  quotientedges(bpc::BeliefPropagationCache)
  partitioned_vertices(bpc::BeliefPropagationCache)
  ```

* Environment around `verts`: incoming BP messages plus the in-partition tensors not in
  `verts`, suitable for local contractions (`caches/beliefpropagationcache.jl`):
  ```julia
  environment(bpc::BeliefPropagationCache, verts::Vector; kwargs...)
  ```

* Scalar associated with a region — for a partition vertex it's the local state contracted
  with incoming messages; for a partition edge it's the inner product of the two messages
  on that bond (`caches/beliefpropagationcache.jl`):
  ```julia
  region_scalar(bp_cache::BeliefPropagationCache, pv::QuotientVertex)
  region_scalar(bp_cache::BeliefPropagationCache, pe::QuotientEdge)
  ```

* Return a copy of the cache with messages on `pes` rescaled to unit norm and
  symmetrized so the bond region scalar is 1 (`caches/beliefpropagationcache.jl`):
  ```julia
  rescale_messages(bp_cache::BeliefPropagationCache, pes)
  ```


