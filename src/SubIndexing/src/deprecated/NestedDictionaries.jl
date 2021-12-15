using Dictionaries
include("SubIndexing.jl")

#struct NestedDictionary
#
#end

d_A = dictionary([(1, 1) => 1, (2, 1) => 2, (1, 2) => 3, (2, 2) => 4])
d_B = dictionary([(1, 1) => 5, (2, 1) => 6, (1, 2) => 7, (2, 2) => 8])
d = dictionary(["A" => d_A, "B" => d_B])

@show d["A"][(1, 1)]
@show d["A"][(2, 1)]
@show d["A"][(1, 2)]
@show d["A"][(2, 2)]
@show d["B"][(1, 1)]
@show d["B"][(2, 1)]
@show d["B"][(1, 2)]
@show d["B"][(2, 2)]

println()
@show d[Sub("A")[(1, 1)]]
@show d[Sub("A")[(2, 1)]]
@show d[Sub("A")[(1, 2)]]
@show d[Sub("A")[(2, 2)]]
@show d[Sub("B")[(1, 1)]]
@show d[Sub("B")[(2, 1)]]
@show d[Sub("B")[(1, 2)]]
@show d[Sub("B")[(2, 2)]]

struct NestedDictionary{SI,T,I1,I2} <: AbstractDictionary{SI,T}
  dicts::Dictionary{I1,Dictionary{I2,T}}
  subindices::Vector{SubIndex{I1,I2}}
end

NestedDictionary(["A" => ["A1", "A2", "A3"], "B" => ["B1", "B2", "B3"]])

subindices(nd::NestedDictionary) = nd.subindices

