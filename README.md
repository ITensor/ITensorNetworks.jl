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



## Examples

Here are is an example of making a tensor network on a chain graph (a tensor train or matrix product state):

```julia
julia> using ITensors

julia> using ITensorNetworks

julia> using Graphs

julia> tn = ITensorNetwork(grid((4,)); link_space=2)
ITensorNetwork with 4 vertices:
4-element Vector{Tuple}:
 (1,)
 (2,)
 (3,)
 (4,)

and 3 edge(s):
(1,) => (2,)
(2,) => (3,)
(3,) => (4,)

with vertex data:
4-element Dictionaries.Dictionary{Tuple, Any}
 (1,) │ ((dim=2|id=829|"1↔2"),)
 (2,) │ ((dim=2|id=829|"1↔2"), (dim=2|id=296|"2↔3"))
 (3,) │ ((dim=2|id=296|"2↔3"), (dim=2|id=274|"3↔4"))
 (4,) │ ((dim=2|id=274|"3↔4"),)

julia> tn[1]
ITensor ord=1 (dim=2|id=829|"1↔2")
NDTensors.EmptyStorage{NDTensors.EmptyNumber, NDTensors.Dense{NDTensors.EmptyNumber, Vector{NDTensors.EmptyNumber}}}

julia> tn[2]
ITensor ord=2 (dim=2|id=829|"1↔2") (dim=2|id=296|"2↔3")
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
julia> tn = ITensorNetwork(grid((2, 2)); dims=(2, 2), link_space=2)
ITensorNetwork with 4 vertices:
4-element Vector{Tuple}:
 (1, 1)
 (2, 1)
 (1, 2)
 (2, 2)

and 4 edge(s):
(1, 1) => (2, 1)
(1, 1) => (1, 2)
(2, 1) => (2, 2)
(1, 2) => (2, 2)

with vertex data:
4-element Dictionaries.Dictionary{Tuple, Any}
 (1, 1) │ ((dim=2|id=334|"1×1↔2×1"), (dim=2|id=475|"1×1↔1×2"))
 (2, 1) │ ((dim=2|id=334|"1×1↔2×1"), (dim=2|id=905|"2×1↔2×2"))
 (1, 2) │ ((dim=2|id=475|"1×1↔1×2"), (dim=2|id=76|"1×2↔2×2"))
 (2, 2) │ ((dim=2|id=905|"2×1↔2×2"), (dim=2|id=76|"1×2↔2×2"))

julia> tn[1, 1]
ITensor ord=2 (dim=2|id=334|"1×1↔2×1") (dim=2|id=475|"1×1↔1×2")
NDTensors.EmptyStorage{NDTensors.EmptyNumber, NDTensors.Dense{NDTensors.EmptyNumber, Vector{NDTensors.EmptyNumber}}}

julia> neighbors(tn, 1, 1)
2-element Vector{Tuple{Int64, Int64}}:
 (2, 1)
 (1, 2)

julia> neighbors(tn, 1, 2)
2-element Vector{Tuple{Int64, Int64}}:
 (1, 1)
 (2, 2)

julia> tn_1 = tn[1, :]
ITensorNetwork with 2 vertices:
2-element Vector{Tuple}:
 (1, 1)
 (1, 2)

and 1 edge(s):
(1, 1) => (1, 2)

with vertex data:
2-element Dictionaries.Dictionary{Tuple, Any}
 (1, 1) │ ((dim=2|id=334|"1×1↔2×1"), (dim=2|id=475|"1×1↔1×2"))
 (1, 2) │ ((dim=2|id=475|"1×1↔1×2"), (dim=2|id=76|"1×2↔2×2"))

julia> tn_2 = tn[2, :]
ITensorNetwork with 2 vertices:
2-element Vector{Tuple}:
 (2, 1)
 (2, 2)

and 1 edge(s):
(2, 1) => (2, 2)

with vertex data:
2-element Dictionaries.Dictionary{Tuple, Any}
 (2, 1) │ ((dim=2|id=334|"1×1↔2×1"), (dim=2|id=905|"2×1↔2×2"))
 (2, 2) │ ((dim=2|id=905|"2×1↔2×2"), (dim=2|id=76|"1×2↔2×2"))
```


Networks can also be merged/unioned:

```julia
julia> tn1 = ITensorNetwork(grid((3,)); vertices=["A", "B", "C"], link_space=2)
ITensorNetwork with 3 vertices:
3-element Vector{Tuple}:
 ("A",)
 ("B",)
 ("C",)

and 2 edge(s):
("A",) => ("B",)
("B",) => ("C",)

with vertex data:
3-element Dictionaries.Dictionary{Tuple, Any}
 ("A",) │ ((dim=2|id=980|"A↔B"),)
 ("B",) │ ((dim=2|id=980|"A↔B"), (dim=2|id=504|"B↔C"))
 ("C",) │ ((dim=2|id=504|"B↔C"),)

julia> tn2 = ITensorNetwork(grid((3,)); vertices=["D", "E", "F"], link_space=2)
ITensorNetwork with 3 vertices:
3-element Vector{Tuple}:
 ("D",)
 ("E",)
 ("F",)

and 2 edge(s):
("D",) => ("E",)
("E",) => ("F",)

with vertex data:
3-element Dictionaries.Dictionary{Tuple, Any}
 ("D",) │ ((dim=2|id=22|"D↔E"),)
 ("E",) │ ((dim=2|id=22|"D↔E"), (dim=2|id=26|"E↔F"))
 ("F",) │ ((dim=2|id=26|"E↔F"),)
```


## Generating this README

This file was generated with [weave.jl](https://github.com/JunoLab/Weave.jl) with the following commands:
```julia
using ITensorNetworks, Weave
weave(joinpath(pkgdir(ITensorNetworks), "examples", "README.jl"); doctype="github", out_path=pkgdir(ITensorNetworks))
```

```julia
```

