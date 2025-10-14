using Test: @test, @testset
using ITensorNetworks: AbstractProblem, default_kwargs, RegionIterator, problem, region_kwargs, @with_defaults

module KwargsTestModule

using ITensorNetworks
using ITensorNetworks: AbstractProblem, @define_default_kwargs

struct TestProblem <: AbstractProblem end
struct NotOurTestProblem <: AbstractProblem end

@define_default_kwargs function test_function(::AbstractProblem; bool=false, int=3)
  return bool, int
end
@define_default_kwargs function test_function(::TestProblem; bool=true, int=0)
  return bool, int
end

end # KwargsTestModule

@testset "Default kwargs" begin
  import .KwargsTestModule

  our_iter = RegionIterator(KwargsTestModule.TestProblem(), ["region" => (; test_function_kwargs=(; int=1))], 1)
  not_our_iter = RegionIterator(KwargsTestModule.NotOurTestProblem(), ["region" => (; test_function_kwargs=(; int=2))], 1)

  kw = region_kwargs(KwargsTestModule.test_function, our_iter)
  @test kw == (; int=1)
  kw_not = region_kwargs(KwargsTestModule.test_function, not_our_iter)
  @test kw_not == (; int=2)

  # Test dispatch
  @test default_kwargs(KwargsTestModule.test_function, problem(our_iter)) == (; bool=true, int=0)

  @test default_kwargs(KwargsTestModule.test_function, problem(not_our_iter)) == (; bool=false, int=3)

  @test KwargsTestModule.test_function(problem(our_iter); default_kwargs(KwargsTestModule.test_function, problem(our_iter); kw...)...) == (true, 1)
  @test KwargsTestModule.test_function(problem(not_our_iter); default_kwargs(KwargsTestModule.test_function, problem(not_our_iter); kw_not...)...) == (false, 2)

  @test @with_defaults(KwargsTestModule.test_function(problem(our_iter))) == (true, 0)
  @test @with_defaults(KwargsTestModule.test_function(problem(our_iter);)) == (true, 0)
  @test @with_defaults(KwargsTestModule.test_function(problem(our_iter); bool = false)) == (false, 0)

  let testval = @with_defaults KwargsTestModule.test_function(problem(our_iter); int = 3)
    @test testval == (true, 3)
  end
end
