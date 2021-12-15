println(
  """
  Test SubIndexing
  """
)

include("SubIndexing.jl")

@show d = Dict("X" => Dict("X1" => 1, "X2" => 2), "Y" => Dict("Y1" => 1, "Y2" => 2))
@show d[Sub("X")["X1"]]

@show d = Dict("X" => randn(2, 2), "Y" => randn(3, 3))
@show d[Sub("X")[1, 2]]

@show d = Dict("X" => [["X1", "X2"], ["X3", "X4"]], "Y" => [["Y1", "Y2"], ["Y3", "Y4"]])
@show d[Sub("X")[1][2]]
@show d[Sub("X")[2][1:2]]
@show d[Sub("X")[2]]
@show d[Sub("X")]

dom = [Sub("A")[1], Sub("A")[2], Sub("B")[1], Sub("B")[2]]

data = ["A" => ["AX1" => "AY1", "AX2" => "AY2"], "B" => ["BX1" => "BY1", "BX2" => "BY2"]]
nested_dict = Dictionary(first.(data), dictionary.(last.(data)))
