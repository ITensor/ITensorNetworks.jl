using Test: @test, @testset
using ITensorNetworks: AbstractProblem, default_kwargs, current_kwargs, RegionIterator, problem

module KwargsTestModule

using ITensorNetworks
using ITensorNetworks: AbstractProblem

export TestProblem, NotOurTestProblem, test_function

struct TestProblem <: AbstractProblem end
struct NotOurTestProblem <: AbstractProblem end

test_function(; bool=false, int=0) = bool, int

function ITensorNetworks.default_kwargs(::typeof(test_function), ::Type{<:AbstractProblem})
  return (; int=3)
end
function ITensorNetworks.default_kwargs(::typeof(test_function), ::Type{<:TestProblem})
  return (; bool=true)
end

end # KwargsTestModule

@testset "Default kwargs" begin
  using .KwargsTestModule: TestProblem, NotOurTestProblem, test_function

  our_iter = RegionIterator(TestProblem(), ["region" => (; int=1)], 1)
  not_our_iter = RegionIterator(NotOurTestProblem(), ["region" => (; int=2)], 1)

  # Test dispatch
  @test default_kwargs(test_function, our_iter) == (; bool=true)
  @test default_kwargs(test_function, problem(our_iter)) == (; bool=true)
  @test default_kwargs(test_function, typeof(problem(our_iter))) == (; bool=true)

  @test default_kwargs(test_function, not_our_iter) == (; int=3)
  @test default_kwargs(test_function, problem(not_our_iter)) == (; int=3)
  @test default_kwargs(test_function, typeof(problem(not_our_iter))) == (; int=3)

  @test test_function(; current_kwargs(test_function, our_iter)...) == (true, 0)
  @test test_function(; current_kwargs(test_function, not_our_iter)...) == (false, 3)

end
