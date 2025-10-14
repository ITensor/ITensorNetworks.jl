using Test: @test, @testset
using ITensorNetworks: SweepIterator, islaststep, state, increment!, compute!, eachregion

module TestIteratorUtils

using ITensorNetworks

struct TestProblem <: ITensorNetworks.AbstractProblem
  data::Vector{Int}
end
ITensorNetworks.region_plan(::TestProblem) = [:a => (; val=1), :b => (; val=2)]
function ITensorNetworks.compute!(iter::ITensorNetworks.RegionIterator{<:TestProblem})
  kwargs = ITensorNetworks.region_kwargs(iter)
  push!(ITensorNetworks.problem(iter).data, kwargs.val)
  return iter
end


mutable struct TestIterator <: ITensorNetworks.AbstractNetworkIterator
  state::Int
  max::Int
  output::Vector{Int}
end

ITensorNetworks.increment!(TI::TestIterator) = TI.state += 1
Base.length(TI::TestIterator) = TI.max
ITensorNetworks.state(TI::TestIterator) = TI.state
function ITensorNetworks.compute!(TI::TestIterator)
  push!(TI.output, ITensorNetworks.state(TI))
  return TI
end

mutable struct SquareAdapter <: ITensorNetworks.AbstractNetworkIterator
  parent::TestIterator
end

Base.length(SA::SquareAdapter) = length(SA.parent)
ITensorNetworks.increment!(SA::SquareAdapter) = ITensorNetworks.increment!(SA.parent)
ITensorNetworks.state(SA::SquareAdapter) = ITensorNetworks.state(SA.parent)
function ITensorNetworks.compute!(SA::SquareAdapter)
  ITensorNetworks.compute!(SA.parent)
  return last(SA.parent.output)^2
end

end

@testset "Iterators" begin

  import .TestIteratorUtils

  @testset "`AbstractNetworkIterator` Interface" begin
    TI = TestIteratorUtils.TestIterator(1, 4, [])

    @test !islaststep((TI))

    # First iterator should compute only
    rv, st = iterate(TI)
    @test !islaststep((TI))
    @test !st
    @test rv === TI
    @test length(TI.output) == 1
    @test only(TI.output) == 1
    @test state(TI) == 1
    @test !st

    rv, st = iterate(TI, st)
    @test !islaststep((TI))
    @test !st
    @test length(TI.output) == 2
    @test state(TI) == 2
    @test TI.output == [1, 2]

    increment!(TI)
    @test !islaststep((TI))
    @test state(TI) == 3
    @test length(TI.output) == 2
    @test TI.output == [1, 2]

    compute!(TI)
    @test !islaststep((TI))
    @test state(TI) == 3
    @test length(TI.output) == 3
    @test TI.output == [1, 2, 3]

    # Final Step
    iterate(TI, false)
    @test islaststep((TI))
    @test state(TI) == 4
    @test length(TI.output) == 4
    @test TI.output == [1, 2, 3, 4]

    @test iterate(TI, false) === nothing

    TI = TestIteratorUtils.TestIterator(1, 5, [])

    cb = []

    for _ in TI
      @test length(cb) == length(TI.output) - 1
      @test cb == (TI.output)[1:end-1]
      push!(cb, state(TI))
      @test cb == TI.output
    end

    @test islaststep((TI))
    @test length(TI.output) == 5
    @test length(cb) == 5
    @test cb == TI.output


    TI = TestIteratorUtils.TestIterator(1, 5, [])
  end

  @testset "Adapters" begin
    TI = TestIteratorUtils.TestIterator(1, 5, [])
    SA = TestIteratorUtils.SquareAdapter(TI)

    @testset "Generic" begin

      i = 0
      for rv in SA
        i += 1
        @test rv isa Int
        @test rv == i^2
        @test state(SA) == i
      end

      @test islaststep((SA))

      TI = TestIteratorUtils.TestIterator(1, 5, [])
      SA = TestIteratorUtils.SquareAdapter(TI)

      SA_c = collect(SA)

      @test SA_c isa Vector
      @test length(SA_c) == 5
      @test SA_c == [1, 4, 9, 16, 25]

    end

    @testset "EachRegion" begin
      prob = TestIteratorUtils.TestProblem([])
      prob_region = TestIteratorUtils.TestProblem([])

      SI = SweepIterator(prob, 5)
      SI_region = SweepIterator(prob_region, 5)

      callback = []
      callback_region = []

      let i = 1
        for _ in SI
          push!(callback, i)
          i += 1
        end
      end

      @test length(callback) == 5

      let i = 1
        for _ in eachregion(SI_region)
          push!(callback_region, i)
          i += 1
        end
      end

      @test length(callback_region) == 10

      @test prob.data == prob_region.data

      @test prob.data[1:2:end] == fill(1, 5)
      @test prob.data[2:2:end] == fill(2, 5)

    end
  end
end
