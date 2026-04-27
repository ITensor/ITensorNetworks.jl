# Experimental Methods

Methods which still need to be discussed, modified, or deprecated.

## ITensorNetwork Methods

* Construct an `ITensorNetwork` from an `IndsNetwork`. Initializes ITensors with `undef` storage on each vertex
  of the `IndsNetwork` with the corresponding indices (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(eltype::Type, undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(eltype::Type, is::IndsNetwork; kwargs...)
  ITensorNetwork(undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(is::IndsNetwork; kwargs...)
  ```

* Extract the `IndsNetwork` of a tensor network — site indices per vertex and link
  indices per edge (`abstractitensornetwork.jl`):
  ```julia
  IndsNetwork(tn::AbstractITensorNetwork)
  ```

* Collect all site indices (per-vertex) of a network as an `IndsNetwork` (`abstractitensornetwork.jl`):
  ```julia
  siteinds(tn::AbstractITensorNetwork)
  ```

* Collect all link indices (per-edge) of a network as an `IndsNetwork` (`abstractitensornetwork.jl`):
  ```julia
  linkinds(tn::AbstractITensorNetwork)
  ```

* Rewrite every index of a network according to a structural mapping `IndsNetwork => IndsNetwork`
  (site indices per vertex, link indices per edge). The two `IndsNetwork`s must share the
  same underlying graph (`abstractitensornetwork.jl`):
  ```julia
  replaceinds(tn::AbstractITensorNetwork, is_is′::Pair{<:IndsNetwork, <:IndsNetwork})
  ```

## IndsNetwork Type and Methods

#### Site Index Helpers

* Build an `IndsNetwork` of site indices on `g` from a value `x` — site-type string,
  dimension, `Index`, or per-vertex dictionary (`sitetype.jl`):
  ```julia
  siteinds(x, g::AbstractGraph; kwargs...)
  ```

* Same, on a length-`nv` path graph (`sitetype.jl`):
  ```julia
  siteinds(x, nv::Int; kwargs...)
  ```

* Build an `IndsNetwork` by calling `f(v)` at each vertex (`sitetype.jl`):
  ```julia
  siteinds(f::Function, g::AbstractGraph; kwargs...)
  ```

#### AbstractIndsNetwork

* Required-to-implement abstract interface — concrete subtypes must define
  `data_graph`; `is_directed` defaults to `false` and may be overloaded (`abstractindsnetwork.jl`):
  ```julia
  data_graph(graph::AbstractIndsNetwork)
  is_directed(::Type{<:AbstractIndsNetwork})
  ```

* Vertex/edge data forwarded from the underlying `DataGraph`, plus the eltype
  declaration `Vector{I}` (`abstractindsnetwork.jl`):
  ```julia
  vertex_data(graph::AbstractIndsNetwork, args...)
  edge_data(graph::AbstractIndsNetwork, args...)
  edge_data_eltype(::Type{<:AbstractIndsNetwork{V, I}}) where {V, I}
  ```

* Indices "unique" to one side of an edge — for `edge`, returns the indices on
  `src(edge)` together with all of its other incident-edge link indices (`abstractindsnetwork.jl`):
  ```julia
  uniqueinds(is::AbstractIndsNetwork, edge::AbstractEdge)
  uniqueinds(is::AbstractIndsNetwork, edge::Pair)
  ```

* Merge two `AbstractIndsNetwork`s, returning an `IndsNetwork` over the merged graph (`abstractindsnetwork.jl`):
  ```julia
  union(is1::AbstractIndsNetwork, is2::AbstractIndsNetwork; kwargs...)
  ```

* Rename every vertex `v` to `f(v)` (`abstractindsnetwork.jl`):
  ```julia
  rename_vertices(f::Function, tn::AbstractIndsNetwork)
  ```

* Promoted index type across all site and link indices in the network (`abstractindsnetwork.jl`):
  ```julia
  promote_indtypeof(is::AbstractIndsNetwork)
  ```

* Build an `IndsNetwork` whose site/link indices at each vertex/edge are the union
  of the corresponding indices from each input network (graphs must match) (`abstractindsnetwork.jl`):
  ```julia
  union_all_inds(is_in::AbstractIndsNetwork...)
  ```

* Insert a default link index on every edge of `indsnetwork` that doesn't already
  have one — `link_space` controls the default bond dimension (`abstractindsnetwork.jl`):
  ```julia
  insert_linkinds(indsnetwork::AbstractIndsNetwork, edges = edges(indsnetwork); link_space = trivial_space(indsnetwork))
  ```

#### IndsNetwork

* Type-parameter accessors and graph-type metadata (`indsnetwork.jl`):
  ```julia
  indtype(inds_network::IndsNetwork)
  indtype(::Type{<:IndsNetwork{V, I}}) where {V, I}
  data_graph(is::IndsNetwork)
  underlying_graph(is::IndsNetwork)
  vertextype(::Type{<:IndsNetwork{V}}) where {V}
  underlying_graph_type(G::Type{<:IndsNetwork})
  is_directed(::Type{<:IndsNetwork})
  ```

* Construct an `IndsNetwork` from a pre-built `DataGraph` (`indsnetwork.jl`):
  ```julia
  IndsNetwork{V, I}(data_graph::DataGraph)
  IndsNetwork{V}(data_graph::DataGraph)
  IndsNetwork(data_graph::DataGraph)
  ```

* Construct from an underlying graph plus link- and site-space specs (positional
  or as `link_space` / `site_space` kwargs). Each spec may be an integer, a `Vector{Int}`,
  an `Index`, a `Vector{<:Index}`, or a per-edge / per-vertex `Dictionary` of any of
  those. `nothing` leaves it empty (`indsnetwork.jl`):
  ```julia
  IndsNetwork{V, I}(g::AbstractNamedGraph, link_space, site_space)
  IndsNetwork{V, I}(g::AbstractSimpleGraph, link_space, site_space)
  IndsNetwork{V}(g, link_space, site_space)
  IndsNetwork(g, link_space, site_space)
  IndsNetwork{V, I}(g; link_space, site_space)
  IndsNetwork{V}(g; link_space, site_space)
  IndsNetwork(g; kwargs...)
  ```

* Core constructor — takes pre-built `Dictionary` link- and site-space maps and
  populates the underlying `DataGraph` directly (`indsnetwork.jl`):
  ```julia
  IndsNetwork{V, I}(g::AbstractNamedGraph, link_space::Dictionary, site_space::Dictionary)
  IndsNetwork{V, I}(g::AbstractSimpleGraph, link_space::Dictionary, site_space::Dictionary)
  ```

* Build an `IndsNetwork` on a path graph from a vector of external (site) indices
  per vertex, or one index per vertex (`indsnetwork.jl`):
  ```julia
  path_indsnetwork(external_inds::Vector{<:Vector{<:Index}})
  path_indsnetwork(external_inds::Vector{<:Index})
  ```

* Normalize a user-supplied link-space spec into a `Dictionary{edgetype, Vector{I}}`,
  building fresh edge-tagged `Index` objects from raw integer dimensions when needed.
  Accepts an integer, a `Vector{Int}`, a `Dictionary` of integers / vectors of integers /
  `Index` / `Vector{<:Index}`, or `nothing` (`indsnetwork.jl`):
  ```julia
  link_space_map(V::Type, I::Type{<:Index}, g, link_space)
  ```

* Same for site spaces, normalizing into `Dictionary{V, Vector{I}}` (`indsnetwork.jl`):
  ```julia
  site_space_map(V::Type, I::Type{<:Index}, g, site_space)
  ```

* Copy an `IndsNetwork` (deep-copies the underlying `DataGraph`) (`indsnetwork.jl`):
  ```julia
  copy(is::IndsNetwork)
  ```

* Apply an index-label transformation `f` to every site index (`sites` kwarg) and/or
  link index (`links` kwarg) of the network (`indsnetwork.jl`):
  ```julia
  map_inds(f, is::IndsNetwork, args...; sites = nothing, links = nothing, kwargs...)
  ```

* Visualize an `IndsNetwork` by wrapping it in a default `ITensorNetwork` (`indsnetwork.jl`):
  ```julia
  visualize(is::IndsNetwork, args...; kwargs...)
  ```
