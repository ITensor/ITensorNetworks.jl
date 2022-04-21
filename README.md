# ITensorNetworks

A package to provide general network data structures and tools to use with ITensors.jl.

## Installation

This package relies on a few unregistered packages. To install, you will need to do:
```julia
julia> using Pkg

julia> Pkg.add(url="https://github.com/mtfishman/MultiDimDictionaries.jl")

julia> Pkg.add(url="https://github.com/mtfishman/NamedGraphs.jl")

julia> Pkg.add(url="https://github.com/mtfishman/DataGraphs.jl")

julia> Pkg.add(url="https://github.com/mtfishman/ITensorNetworks.jl")
```

Here are is an example of making a tensor network on a chain graph (a tensor train or matrix product state):
```julia
julia> using ITensors, ITensorNetworks, Graphs

julia> tn = ITensorNetwork(grid((4,)))
ITensorNetwork with 4 tensors
Network partitions:
Partition("") => [1, 2, 3, 4]

and 3 edges:
1 => 2
2 => 3
3 => 4


julia> tn[1]
ITensor ord=1 (dim=1|id=466|"l=1→2")
NDTensors.EmptyStorage{NDTensors.EmptyNumber, NDTensors.Dense{NDTensors.EmptyNumber, Vector{NDTensors.EmptyNumber}}}

julia> tn[2]
ITensor ord=2 (dim=1|id=466|"l=1→2") (dim=1|id=249|"l=2→3")
NDTensors.EmptyStorage{NDTensors.EmptyNumber, NDTensors.Dense{NDTensors.EmptyNumber, Vector{NDTensors.EmptyNumber}}}

julia> neighbors(tn, 1)
1-element Vector{Int64}:
 2

julia> neighbors(tn, 2)
2-element Vector{Int64}:
 1
 3

julia> neighbors(tn, 3)
2-element Vector{Int64}:
 2
 4

julia> neighbors(tn, 4)
1-element Vector{Int64}:
 3

```
and here is a similar example for making a tensor network on a grid (a tensor product state or project entangled pair state (PEPS)):
```julia
julia> tn = ITensorNetwork(grid((2, 2)), (2, 2))
ITensorNetwork with 4 tensors
Network partitions:
Partition("") => [(1, 1), (2, 1), (1, 2), (2, 2)]

and 4 edges:
(1, 1) => (2, 1)
(1, 1) => (1, 2)
(2, 1) => (2, 2)
(1, 2) => (2, 2)


julia> tn[(1, 1)]
ITensor ord=2 (dim=1|id=325|"l=1→2") (dim=1|id=697|"l=1→3")
NDTensors.EmptyStorage{NDTensors.EmptyNumber, NDTensors.Dense{NDTensors.EmptyNumber, Vector{NDTensors.EmptyNumber}}}

julia> neighbors(tn, (1, 1))
2-element Vector{Tuple{Int64, Int64}}:
 (2, 1)
 (1, 2)

julia> neighbors(tn, (1, 2))
2-element Vector{Tuple{Int64, Int64}}:
 (1, 1)
 (2, 2)

```

Networks can also be produced together:
```julia
julia> tn1 = ITensorNetwork(grid((3,)), ["A", "B", "C"])
ITensorNetwork with 3 tensors
Network partitions:
Partition("") => ["A", "B", "C"]

and 2 edges:
"A" => "B"
"B" => "C"


julia> tn2 = ITensorNetwork(grid((3,)), ["D", "E", "F"])
ITensorNetwork with 3 tensors
Network partitions:
Partition("") => ["D", "E", "F"]

and 2 edges:
"D" => "E"
"E" => "F"


julia> tn1 ⊗ tn2
ITensorNetwork with 6 tensors
Network partitions:
Partition("") => ["A", "B", "C", "D", "E", "F"]

and 4 edges:
"A" => "B"
"B" => "C"
"D" => "E"
"E" => "F"

```
