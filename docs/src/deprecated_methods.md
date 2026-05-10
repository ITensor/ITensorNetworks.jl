# Deprecated Methods

Suggestions of methods which could be deleted.

## Global Operations on ITensorNetworks


## Miscellaneous Methods

* Methods in `partitioneditensornetwork.jl`.
  ```julia
  linkinds(pitn::PartitionedGraph, edge::QuotientEdge)
  ```
  To be revisited after Jack's work on NamedGraphs.

* "Split" an edge index by applying a map to each copy of it on the adjacent
  ITensors. By default the `dst(edge)` copy is primed and the `src(edge)`
  copy is unchanged (`abstractitensornetwork.jl`):
  ```julia
  split_index(tn::AbstractITensorNetwork, edges_to_split;
              src_ind_map::Function = identity,
              dst_ind_map::Function = prime)
  ```
  To be removed alongside the `insert_linkinds` / `add_edge!` /
  `factorize_edge!` redesign so the public edge-mutation API surface
  lands as a single coherent set.

