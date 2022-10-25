using TimerOutputs
using ITensorNetworks.ApproximateTNContraction:
  timer, merge_tree, subtree, vectorize, topo_sort, get_leaves

@testset "test merge tree" begin
  t1 = [[1], [2], [3]]
  t2 = [4, 5, 6]
  @test merge_tree(t1, t2; append=true) == [[1], [2], [3], [4, 5, 6]]
  @test merge_tree(t1, t2; append=false) == [[[1], [2], [3]], [4, 5, 6]]
  @test merge_tree([], [1, 2, 3]; append=false) == [1, 2, 3]
end

@testset "test subtree and vectorize" begin
  t1 = [[[1, 2], [3]], [4]]
  subset = [1]
  @test subtree(t1, subset) == [1]
  @test vectorize(t1) == [1, 2, 3, 4]
end

@testset "test find topo sort" begin
  reset_timer!(timer)
  tn = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  @test length(topo_sort(tn)) == 7
  @test length(topo_sort(tn; leaves=get_leaves(tn))) == 3
  show(timer)
end
