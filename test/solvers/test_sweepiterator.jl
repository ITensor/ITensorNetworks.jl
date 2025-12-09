using Test: @test, @testset
using ITensorNetworks: ITensorNetworks, AbstractProblem, RegionIterator, SweepIterator, compute!, region_iterator, region_kwargs

include("utilities/tree_graphs.jl")

# TestProblem type for testing
struct TestProblem <: AbstractProblem
    graph
end

ITensorNetworks.state(T::TestProblem) = T.graph

ITensorNetworks.compute!(R::RegionIterator{<:TestProblem}) = "TestProblem Compute"


@testset "SweepIterator Basics" begin
    g = build_tree(; nbranch = 3, nbranch_sites = 3)
    prob = TestProblem(g)

    nsweeps = 5

    # Basic construction, taking length
    sweep_iter = SweepIterator(prob, nsweeps)
    @test length(sweep_iter) == nsweeps

    # Pass keyword parameters
    test_kwarg_a = 1
    test_kwarg_b = "b"
    sweep_iter = SweepIterator(prob, nsweeps; test_kwarg_a, test_kwarg_b)
    @test region_kwargs(region_iterator(sweep_iter)).test_kwarg_a == test_kwarg_a
    @test region_kwargs(region_iterator(sweep_iter)).test_kwarg_b == test_kwarg_b

    # Pass array of parameters
    kws_array = [(; outputlevel = 0), (; outputlevel = 1)]
    sweep_iter = SweepIterator(prob, kws_array)
    @test length(sweep_iter) == length(kws_array)
    @test region_kwargs(region_iterator(sweep_iter)).outputlevel == 0
end

@testset "SweepIterator Iteration" begin
    g = build_tree(; nbranch = 3, nbranch_sites = 3)
    prob = TestProblem(g)

    nsweeps = 5
    sweep_iter = SweepIterator(prob, nsweeps)
    count = 0
    for _ in sweep_iter
        count += 1
    end
    @test count == nsweeps

    # Test case of one iteration
    nsweeps = 1
    sweep_iter = SweepIterator(prob, nsweeps)
    count = 0
    for _ in sweep_iter
        count += 1
    end
    @test count == nsweeps
end
