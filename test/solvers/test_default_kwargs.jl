using Test: @test, @testset
using ITensorNetworks: AbstractProblem, default_kwargs, RegionIterator, problem, region_kwargs

module KwargsTestModule

using ITensorNetworks
using ITensorNetworks: AbstractProblem, @default_kwargs

export TestProblem, NotOurTestProblem, test_function

struct TestProblem <: AbstractProblem end
struct NotOurTestProblem <: AbstractProblem end

@default_kwargs astypes = true function test_function(::AbstractProblem; bool=false, int=3)
  return bool, int
end
@default_kwargs astypes = true function test_function(::TestProblem; bool=true, int=0)
  return bool, int
end

end # KwargsTestModule

@testset "Default kwargs" begin
  using .KwargsTestModule: TestProblem, NotOurTestProblem, test_function

  our_iter = RegionIterator(TestProblem(), ["region" => (; test_function_kwargs=(; int=1))], 1)
  not_our_iter = RegionIterator(NotOurTestProblem(), ["region" => (; test_function_kwargs=(; int=2))], 1)

  kw = region_kwargs(test_function, our_iter)
  @test kw == (; int=1)
  kw_not = region_kwargs(test_function, not_our_iter)
  @test kw_not == (; int=2)

  @info methods(default_kwargs)

  # Test dispatch
  @test default_kwargs(test_function, problem(our_iter)) == (; bool=true, int=0)
  @test default_kwargs(test_function, problem(our_iter) |> typeof) == (; bool=true, int=0)

  @test default_kwargs(test_function, problem(not_our_iter)) == (; bool=false, int=3)
  @test default_kwargs(test_function, problem(not_our_iter) |> typeof) == (; bool=false, int=3)

  @test test_function(problem(our_iter); default_kwargs(test_function, problem(our_iter); kw...)...) == (true, 1)
  @test test_function(problem(not_our_iter); default_kwargs(test_function, problem(not_our_iter); kw_not...)...) == (false, 2)

end
