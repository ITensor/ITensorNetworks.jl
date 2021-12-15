println(
  """

  Test AbstractBijections
  """
)

include(joinpath("..", "src", "AbstractBijections.jl"))
using .AbstractBijections

@show f = Bijection(["X1", "X2"], ["Y1", "Y2"])
@show f["X1"]
@show inv(f)["Y1"]
@show domain(f)
@show image(f)

@show f = bijection(["X1" => "Y1", "X2" => "Y2"])
@show f["X1"] == "Y1"
@show inv(f)["Y1"] == "X1"

@show f = Bijection(["Y1", "Y2"])
@show f[1] == "Y1"
@show inv(f)["Y1"] == 1
