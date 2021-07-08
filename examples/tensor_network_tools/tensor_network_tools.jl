using ITensors

abstract type AbstractLattice end

struct IrregularLattice <: AbstractLattice end

struct HyperCubic{N} <: AbstractLattice
  dims::NTuple{N,Int}
end

struct ITensorNetwork{T,L}
  data::Dict{T,ITensor} # Map tensor labels to tensors
  tags::Dict{T,TagSet} # Map tensor labels to tags that categorize the tensors, such as boundary
  lattice::L
  out_boundary::Dict{T,T} # Represent links across boundaries
  in_boundary::Dict{T,T} # Represent links across boundaries
end

ts = ("i", "j", "k", "l")
ds = (2, 2, 2, 2)
i, j, k, l = Index.(ds, ts)
A, B, C, D = ITensor.(((i, j), (j, k), (k, l), (l, i)))

tn = ITensorNetwork(["A" => A, "B" => B, "C" => C, "D" => D])

