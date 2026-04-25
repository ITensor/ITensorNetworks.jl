# Interface Methods

Recommended methods for building applications on top of ITensorNetworks.

## ITensorNetwork Constructors

These ITensorNetwork constructor interfaces are foundational to other constructors:

* Default constructor and copy constructor.
  ```julia
  ITensorNetwork{V}()
  ITensorNetwork{V}(tn::ITensorNetwork)
  ```

* From vertex-tensor pairings.
  ```julia
  # Dictionary of vertices => tensors
  ITensorNetwork(ts::AbstractDictionary{<:Any, ITensor})
  ITensorNetwork(ts::AbstractDict{<:Any, ITensor})

  # Vector of `vertex => ITensor` pairs
  ITensorNetwork(ts::AbstractVector{<:Pair{<:Any, ITensor}})

  # Vector of vertices, vector of ITensors
  ITensorNetwork(vertices::AbstractVector, tensors::AbstractVector{ITensor})
  ```

* From a collection of ITensorNetworks. Merges (Kronecker or tensor product) of input networks.
  ```julia
  ITensorNetwork(itns::Vector{ITensorNetwork})
  ```

* From `IndsNetwork`. Initializes ITensors with `undef` storage on each vertex
  of the `IndsNetwork` with the corresponding indices.
  ```julia
  ITensorNetwork(eltype::Type, undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(eltype::Type, is::IndsNetwork; kwargs...)
  ITensorNetwork(undef::UndefInitializer, is::IndsNetwork; kwargs...)
  ITensorNetwork(is::IndsNetwork; kwargs...)
  ```

## Analyzing ITensorNetworks

* Indices common to the ITensors on the vertices connected by the edge.
  ```julia
  commoninds(tn::AbstractITensorNetwork, edge)
  linkinds(tn::AbstractITensorNetwork, edge)
  ```

* Collection of tensors neighboring the given vertex.
  ```julia
  neighbor_tensors(tn::AbstractITensorNetwork, vertex)
  ```

* Indices on the source tensor of `edge` that are not shared with the destination tensor.
  For `edge = (v1 => v2)`, returns every index of `tn[v1]` except the bond(s) to `tn[v2]`, that is,
  `v1`'s site index plus its link indices to all of `v1`'s other neighbors. Useful for QR/SVD
  across that bond: these are the indices you keep on the `v1` side.
  ```julia
  uniqueinds(tn::AbstractITensorNetwork, edge::AbstractEdge)
  uniqueinds(tn::AbstractITensorNetwork, edge::Pair)
  # Alias for `uniqueinds`.
  siteinds(tn::AbstractITensorNetwork, edge)
  ```

* Indices on `tn[vertex]` that aren't shared with any neighbor, i.e. the external/site
  indices of that vertex.
  ```julia
  uniqueinds(tn::AbstractITensorNetwork, vertex)
  # Alias for `uniqueinds`.
  siteinds(tn::AbstractITensorNetwork, vertex)
  ```

* Tags on the link index (or indices) associated with `edge`.
  ```julia
  tags(tn::AbstractITensorNetwork, edge)
  ```

## Local Operations on ITensorNetworks

* Contract the tensors at vertices `src(edge)` and `dst(edge)` and store the result in
  `merged_vertex` (which defaults to `dst(edge)`), removing the other vertex (defaults to `src(edge)`).
  ```julia
  contract(tn::AbstractITensorNetwork, edge::AbstractEdge; merged_vertex = dst(edge))
  contract(tn::AbstractITensorNetwork, edge::Pair; kws...)
  ```

* Factorize the bond on `edge` using the default factorization.
  ```julia
  factorize(tn::AbstractITensorNetwork, edge::AbstractEdge; kws...)
  factorize(tn::AbstractITensorNetwork, edge::Pair; kws...)
  ```

* QR-factorize across `edge`, placing the orthogonal factor on `src(edge)` and the remainder
  on `dst(edge)`.
  ```julia
  qr(tn::AbstractITensorNetwork, edge::AbstractEdge; kws...)
  ```

* SVD-factorize across `edge`.
  ```julia
  svd(tn::AbstractITensorNetwork, edge::AbstractEdge; kws...)
  ```

* Truncate the bond on `edge` via SVD; forwards `cutoff`, `maxdim`, `mindim` kwargs.
  ```julia
  truncate(tn::AbstractITensorNetwork, edge::AbstractEdge; kws...)
  ```

## Global Operations on ITensorNetworks

* Tensor product (disjoint union) of two ITensorNetworks.
  ```julia
  ⊗(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
  ```

* Elementwise complex conjugation of every tensor in the network.
  ```julia
  conj(tn::AbstractITensorNetwork)
  ```

* Dagger: conjugate every tensor and prime the appropriate indices.
  ```julia
  dag(tn::AbstractITensorNetwork)
  ```

* Approximate equality of two ITensorNetworks.
  ```julia
  isapprox(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kws...)
  ```
