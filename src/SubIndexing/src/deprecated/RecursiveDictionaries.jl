using Dictionaries
include("SubIndexing.jl")

#struct NestedDictionary
#
#end

d = dictionary(["A" => dictionary(["A1" => [1, 2], "A2" => [3, 4]]), "B" => dictionary(["B1" => [5, 6], "B2" => [7, 8]])])

@show d["A"]["A1"][1]
@show d["A"]["A1"][2]
@show d["A"]["A2"][1]
@show d["A"]["A2"][2]
@show d["B"]["B1"][1]
@show d["B"]["B1"][2]
@show d["B"]["B2"][1]
@show d["B"]["B2"][2]

@show d[Sub("A")["A1"][1]]
@show d[Sub("A")["A1"][2]]
@show d[Sub("A")["A2"][1]]
@show d[Sub("A")["A2"][2]]
@show d[Sub("B")["B1"][1]]
@show d[Sub("B")["B1"][2]]
@show d[Sub("B")["B2"][1]]
@show d[Sub("B")["B2"][2]]

abstract type IsCollection end
struct Collection <: IsCollection end
struct Element <: IsCollection end

iscollection(::AbstractDictionary) = Collection()
iscollection(::AbstractArray) = Collection()
iscollection(::AbstractString) = Element()
iscollection(::Number) = Element()

each_collection_index(c) = each_collection_index(iscollection(c), c)
# TODO: Use CartesianIndices for AbstactArray?
each_collection_index(::Collection, c) = eachindex(c)
each_collection_index(::Element, c) = ()

#function eachsubindex(d)
#  for k1 in each_collection_index(d)
#    @show k1
#    @show eachsubindex(d[k1])
#  end
#end

function eachsubindex!(subindices, d, depth)
  for k in each_collection_index(d)
    push!(subindices, k)
    subindices, depth = eachsubindex!(subindices, d[k], depth)
  end
  return subindices, depth + 1
end

function eachsubindex(d)
  subindices = []
  depth = 0
  subindices, depth = eachsubindex!(subindices, d, depth)
  @show depth
  return subindices
end
