println(
  """

  Test SubIndexing
  """
)

include(joinpath("..", "src", "SubIndexing.jl"))
using .SubIndexing

@show Sub("A") isa Sub{String}
@show Sub("A")[1] isa SubIndexing.SubIndex{String,Tuple{Int}}
@show Sub("A") ⊆ Sub("A")
@show Sub("A")[1] ⊆ Sub("A")
@show Sub("A")["X"][1] ⊆ Sub("A")["X"]
@show Sub("A")["X"] ⊈ Sub("A")["Y"]
@show Sub("A")["X"] ⊈ Sub("A")["X"][1]
