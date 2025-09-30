using Test: @test, @testset
using ITensorNetworks: done, state, increment!, compute!

module TestIteratorUtils

using ITensorNetworks

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

  using .TestIteratorUtils: TestIterator, SquareAdapter

  @testset "`AbstractNetworkIterator` Interface" begin
    TI = TestIterator(1, 4, [])

    @test !done(TI)

    # First iterator should compute only
    rv, st = iterate(TI)
    @test !done(TI)
    @test !st
    @test rv === TI
    @test length(TI.output) == 1
    @test only(TI.output) == 1
    @test state(TI) == 1
    @test !st

    rv, st = iterate(TI, st)
    @test !done(TI)
    @test !st
    @test length(TI.output) == 2
    @test state(TI) == 2
    @test TI.output == [1, 2]

    increment!(TI)
    @test !done(TI)
    @test state(TI) == 3
    @test length(TI.output) == 2
    @test TI.output == [1, 2]

    compute!(TI)
    @test !done(TI)
    @test state(TI) == 3
    @test length(TI.output) == 3
    @test TI.output == [1, 2, 3]

    # Final Step
    iterate(TI, false)
    @test done(TI)
    @test state(TI) == 4
    @test length(TI.output) == 4
    @test TI.output == [1, 2, 3, 4]

    @test iterate(TI, false) === nothing

    TI = TestIterator(1, 5, [])

    cb = []

    for _ in TI
      @test length(cb) == length(TI.output) - 1
      @test cb == (TI.output)[1:end-1]
      push!(cb, state(TI))
      @test cb == TI.output
    end

    @test done(TI)
    @test length(TI.output) == 5
    @test length(cb) == 5
    @test cb == TI.output


    TI = TestIterator(1, 5, [])
  end

  @testset "Adapters" begin
    TI = TestIterator(1, 5, [])
    SA = SquareAdapter(TI)

    i = 0
    for rv in SA
      i += 1
      @test rv isa Int
      @test rv == i^2
      @test state(SA) == i
    end

    @test done(SA)

    TI = TestIterator(1, 5, [])
    SA = SquareAdapter(TI)

    SA_c = collect(SA)

    @test SA_c isa Vector
    @test length(SA_c) == 5
    @test SA_c == [1, 4, 9, 16, 25]
  end
end
