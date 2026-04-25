# Internal Methods

Developer-focused methods for internal use by other parts of ITensorNetworks.

## Local Operations on ITensorNetworks

* Gauge a single edge of an ITensorNetwork, moving the orthogonality/gauge across the bond.
  ```julia
  gauge_edge(tn::AbstractITensorNetwork, edge::AbstractEdge)
  ```

## Global Operations on ITensorNetworks

* Apply a sequence of edge gauges to walk the gauge center along the given edges.
  ```julia
  gauge_walk(alg::Algorithm, tn::AbstractITensorNetwork, edges::Vector{<:AbstractEdge}; kws...)
  gauge_walk(alg::Algorithm, tn::AbstractITensorNetwork, edge::Pair; kws...)
  gauge_walk(alg::Algorithm, tn::AbstractITensorNetwork, edges::Vector{<:Pair}; kws...)
  ```

* Gauge an ITensorNetwork towards a region, treating the network as a tree spanned by a
  spanning tree.
  ```julia
  tree_gauge(alg::Algorithm, ψ::AbstractITensorNetwork, cur_region::Vector, new_region::Vector; kws...)
  tree_gauge(alg::Algorithm, ψ::AbstractITensorNetwork, region)
  tree_gauge(alg::Algorithm, ψ::AbstractITensorNetwork, region::Vector)
  ```

* Orthogonalize an ITensorNetwork towards a region along the spanning tree.
  ```julia
  tree_orthogonalize(ψ::AbstractITensorNetwork, cur_region, new_region; kwargs...)
  tree_orthogonalize(ψ::AbstractITensorNetwork, region; kwargs...)
  ```
