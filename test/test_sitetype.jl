@eval module $(gensym())
using DataGraphs: vertex_data
using Dictionaries: Dictionary
using Graphs: nv, vertices
using ITensorNetworks: IndsNetwork, siteinds
using ITensors: Index, SiteType, hastags, space
using ITensors.NDTensors: dim
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @testset

@testset "Site ind system" begin
  g = named_grid((2, 2))
  sitetypes = [rand(["S=1/2", "S=1", "Boson", "Fermion"]) for _ in 1:nv(g)]
  dims = space.(SiteType.(sitetypes))
  typemap = Dictionary(vertices(g), sitetypes)
  dimmap = Dictionary(vertices(g), dims)
  ftype(v::Tuple) = iseven(sum(isodd.(v))) ? "S=1/2" : "S=1"
  fdim(v::Tuple) = space(SiteType(ftype(v)))
  testtag = "TestTag"

  d1 = map(v -> Index(2), vertices(g))
  d2 = map(v -> "S=1/2", vertices(g))
  for x in (v -> d1[v], d1, v -> d2[v], d2)
    s = siteinds(x, g)
    @test s[1, 1] isa Vector{<:Index}
    @test s[1, 2] isa Vector{<:Index}
    @test s[2, 1] isa Vector{<:Index}
    @test s[2, 2] isa Vector{<:Index}
  end

  # uniform string sitetype
  s_us = siteinds(sitetypes[1], g; addtags=testtag)
  @test s_us isa IndsNetwork
  @test all(dim.(vertex_data(s_us)) .== dims[1])
  @test all(hastags.(vertex_data(s_us), Ref("$testtag,$(sitetypes[1]),Site")))
  # dictionary string sitetype
  s_ds = siteinds(typemap, g; addtags=testtag)
  @test s_ds isa IndsNetwork
  @test all(dim(only(s_ds[v])) == dimmap[v] for v in vertices(g))
  @test all(hastags(only(s_ds[v]), "$testtag,$(typemap[v]),Site") for v in vertices(g))

  # uniform integer sitetype
  s_ui = siteinds(dims[1], g; addtags=testtag)
  @test s_ui isa IndsNetwork
  @test all(dim.(vertex_data(s_ui)) .== dims[1])
  @test all(hastags.(vertex_data(s_ui), Ref("$testtag,Site")))
  # dictionary integer sitetype
  s_di = siteinds(dimmap, g; addtags=testtag)
  @test s_di isa IndsNetwork
  @test all(dim(only(s_di[v])) == dimmap[v] for v in vertices(g))
  @test all(hastags.(vertex_data(s_di), Ref("$testtag,Site")))

  # function string site type
  s_fs = siteinds(ftype, g; addtags=testtag)
  @test s_fs isa IndsNetwork
  @test all(dim(only(s_fs[v])) == fdim(v) for v in vertices(g))
  @test all(hastags(only(s_fs[v]), "$testtag,$(ftype(v)),Site") for v in vertices(g))
  # function integer site type
  s_fi = siteinds(fdim, g; addtags=testtag)
  @test s_fs isa IndsNetwork
  @test all(dim(only(s_fs[v])) == fdim(v) for v in vertices(g))
  @test all(hastags.(vertex_data(s_fs), Ref("$testtag,Site")))
end
end
