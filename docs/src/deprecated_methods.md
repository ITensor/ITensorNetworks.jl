# Deprecated Methods

Suggestions of methods which could be deleted.

## ITensorNetwork Methods

#### ITensorNetwork Constructors

* Default constructor (`itensornetwork.jl`).
  ```julia
  ITensorNetwork{V}()
  ```

* From a named graph, forwards to construction from `IndsNetwork` (`itensornetwork.jl`):
  ```julia
  ITensorNetwork{V}(g::NamedGraph)
  ITensorNetwork(eltype::Type, undef::UndefInitializer, graph::AbstractNamedGraph; kws...)
  ITensorNetwork(f, graph::AbstractNamedGraph; kwargs...)
  ITensorNetwork(graph::AbstractNamedGraph; kwargs...)
  ```

* From a simple graph, forwards to construction from `IndsNetwork` (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(eltype::Type, undef::UndefInitializer, graph::AbstractSimpleGraph; kws...)
  ITensorNetwork(f, graph::AbstractSimpleGraph; kwargs...)
  ITensorNetwork(graph::AbstractSimpleGraph; kwargs...)
  ```

* From a function over vertices or from a "value" (e.g. a string like `"Up"`,
  an `Op`, an array, or a per-vertex dict/array) that is converted to a callable and used
  to initialize each vertex tensor (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(value, is::IndsNetwork; kwargs...)
  ITensorNetwork(elt::Type, f, is::IndsNetwork; link_space = trivial_space(is), kws...)
  ITensorNetwork(itensor_constructor::Function, is::IndsNetwork; link_space = trivial_space(is), kwargs...)
  ```

* From a single `ITensor`. Wraps the tensor in a single-vertex network (`itensornetwork.jl`):
  ```julia
  ITensorNetwork(t::ITensor)
  ```

#### Local Operations on ITensorNetworks

* Versions of `siteinds` taking a `vertex` argument. Each of these is just an alias for `uniqueinds`. Possibly the wrong design / implementation. (`abstractitensornetwork.jl`).
  ```julia
  siteinds(tn::AbstractITensorNetwork, vertex) # abstractitensornetwork.jl:288
  siteinds(tn::AbstractITensorNetwork, vertex::Int) # abstractitensornetwork.jl:292
  ```


* Functions in `apply.jl` which are unused, even inside that file (`apply.jl`):
  ```julia
  _gate_vertices(o::ITensor, ψ)
  _gate_vertices(o::AbstractEdge, ψ)
  _contract_gate(o::ITensor, ψv1, Λ, ψv2)
  _contract_gate(o::AbstractEdge, ψv1, Λ, ψv2)
  ```

* Collection of tensors neighboring the given vertex (`abstractitensornetwork.jl`):
  ```julia
  neighbor_tensors(tn::AbstractITensorNetwork, vertex)
  ```

* Iterate over the tensors at the given vertices, default all vertices (`abstractitensornetwork.jl`):
  ```julia
  eachtensor(tn::AbstractITensorNetwork, vertices = vertices(tn))
  ```

## Global Operations on ITensorNetworks


## TreeTensorNetwork Constructors

* From `Op` and related types (`opsum_to_ttn.jl`):
  ```julia
  ttn(o::Op, s::IndsNetwork; kws...)
  ttn(o::Scaled{C, Op}, s::IndsNetwork; kws...)
  ttn(o::Sum{Op}, s::IndsNetwork; kws...)
  ttn(o::Prod{Op}, s::IndsNetwork; kws...)
  ttn(o::Scaled{C, Prod{Op}}, s::IndsNetwork; kws...)
  ttn(o::Sum{Scaled{C, Op}}, s::IndsNetwork; kws...)
  ```

## Miscellaneous Methods

* Methods in `partitioneditensornetwork.jl`.
  ```julia
  linkinds(pitn::PartitionedGraph, edge::QuotientEdge)
  ```
  To be revisited after Jack's work on NamedGraphs.

* Methods in `graphs.jl`. 
  Just one methods which constructs a `SimpleGraph` from ITensors (`graphs.jl`).
  ```julia
  SimpleGraphs.SimpleGraph(itensors::Vector{ITensor})
  ```
  Not used anywhere in library.

* Methods in `update_observer.jl`. Not used anywhere in library.
