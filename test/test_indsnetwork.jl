@eval module $(gensym())
using DataGraphs: edge_data, vertex_data
using Dictionaries: Dictionary
using Graphs: edges, ne, nv, vertices
using ITensorNetworks: IndsNetwork, union_all_inds
using ITensors: Index
using ITensors.NDTensors: dim
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "IndsNetwork constructors" begin
  # test on comb tree
  dims = (3, 2)
  c = named_comb_tree(dims)
  ## specify some site and link indices in different ways
  # one index per site
  rng = StableRNG(1234)
  site_dims = [rand(rng, 2:6) for _ in 1:nv(c)]
  site_inds = Index.(site_dims)
  # multiple indices per site
  site_dims_multi = [[rand(rng, 2:6) for ni in 1:rand(rng, 1:3)] for _ in 1:nv(c)]
  site_inds_multi = map(x -> Index.(x), site_dims_multi)
  # one index per link
  link_dims = [rand(rng, 2:6) for _ in 1:ne(c)]
  link_inds = Index.(link_dims)
  # multiple indices per link
  link_dims_multi = [[rand(rng, 2:6) for ni in 1:rand(rng, 1:3)] for _ in 1:ne(c)]
  link_inds_multi = map(x -> Index.(x), link_dims_multi)
  # TODO: fix ambiguity due to vectors of QNBlocks...
  # test constructors
  ## empty constructor
  is_emtpy = IndsNetwork(c)
  @test is_emtpy isa IndsNetwork
  @test isempty(vertex_data(is_emtpy)) && isempty(edge_data(is_emtpy))
  ## specify site and/or link spaces uniformly
  uniform_dim = rand(rng, 2:6)
  uniform_dim_multi = [rand(rng, 2:6) for _ in 1:rand(rng, 2:4)]
  # only initialize sites
  is_usite = IndsNetwork(c; site_space=uniform_dim)
  @test is_usite isa IndsNetwork
  @test all(map(x -> dim.(x) == [uniform_dim], vertex_data(is_usite)))
  @test isempty(edge_data(is_usite))
  is_umsite = IndsNetwork(c; site_space=uniform_dim_multi)
  @test is_umsite isa IndsNetwork
  @test all(map(x -> dim.(x) == uniform_dim_multi, vertex_data(is_umsite)))
  @test isempty(edge_data(is_umsite))
  # only initialize links
  is_ulink = IndsNetwork(c; link_space=uniform_dim)
  @test is_ulink isa IndsNetwork
  @test all(map(x -> dim.(x) == [uniform_dim], edge_data(is_ulink)))
  @test isempty(vertex_data(is_ulink))
  is_umlink = IndsNetwork(c; link_space=uniform_dim_multi)
  @test is_umlink isa IndsNetwork
  @test all(map(x -> dim.(x) == uniform_dim_multi, edge_data(is_umlink)))
  @test isempty(vertex_data(is_umlink))
  # initialize sites and links
  is_usite_umlink = IndsNetwork(c; site_space=uniform_dim, link_space=uniform_dim_multi)
  @test is_usite_umlink isa IndsNetwork
  @test all(map(x -> dim.(x) == [uniform_dim], vertex_data(is_usite_umlink)))
  @test all(map(x -> dim.(x) == uniform_dim_multi, edge_data(is_usite_umlink)))

  # specify site spaces as Dictionary of dims or indices, and/or link spaces uniformly
  site_dim_map = Dictionary(vertices(c), site_dims)
  site_inds_map = Dictionary(vertices(c), site_inds)
  site_dim_map_multi = Dictionary(vertices(c), site_dims_multi)
  site_inds_map_multi = Dictionary(vertices(c), site_inds_multi)
  # integer site dimensions, no links
  is_site = IndsNetwork(c; site_space=site_dim_map)
  @test is_site isa IndsNetwork
  @test all(dim.(is_site[v]) == [site_dim_map[v]] for v in vertices(is_site))
  @test isempty(edge_data(is_site))
  is_msite = IndsNetwork(c; site_space=site_dim_map_multi)
  @test is_msite isa IndsNetwork
  @test all(dim.(is_msite[v]) == site_dim_map_multi[v] for v in vertices(is_msite))
  @test isempty(edge_data(is_msite))
  # integer site vector, uniform integer links
  is_msite_ulink = IndsNetwork(c; site_space=site_dim_map_multi, link_space=uniform_dim)
  @test is_msite_ulink isa IndsNetwork
  @test all(
    dim.(is_msite_ulink[v]) == site_dim_map_multi[v] for v in vertices(is_msite_ulink)
  )
  @test all(map(x -> dim.(x) == [uniform_dim], edge_data(is_msite_ulink)))
  # Index site map, no links
  is_isite = IndsNetwork(c; site_space=site_inds_map)
  @test is_isite isa IndsNetwork
  @test all(is_isite[v] == [site_inds_map[v]] for v in vertices(is_isite))
  @test isempty(edge_data(is_isite))
  is_misite = IndsNetwork(c; site_space=site_inds_map_multi)
  @test is_misite isa IndsNetwork
  @test all(is_misite[v] == site_inds_map_multi[v] for v in vertices(is_misite))
  @test isempty(edge_data(is_misite))
  # Index site vector, uniform integer links
  is_misite_ulink = IndsNetwork(c; site_space=site_inds_map_multi, link_space=uniform_dim)
  @test is_misite_ulink isa IndsNetwork
  @test all(is_misite_ulink[v] == site_inds_map_multi[v] for v in vertices(is_misite_ulink))
  @test all(map(x -> dim.(x) == [uniform_dim], edge_data(is_misite_ulink)))

  # specify site spaces as Dictionary of dims or indices, and/or link spaces as Dictionary
  # of indices
  link_dim_map = Dictionary(edges(c), link_dims)
  link_inds_map = Dictionary(edges(c), link_inds)
  link_dim_map_multi = Dictionary(edges(c), link_dims_multi)
  link_inds_map_multi = Dictionary(edges(c), link_inds_multi)
  # index site dict, integer link dict
  is_isite_link = IndsNetwork(c; site_space=site_inds_map, link_space=link_dim_map)
  @test is_isite_link isa IndsNetwork
  @test all(is_isite_link[v] == [site_inds_map[v]] for v in vertices(is_isite_link))
  @test all(dim.(is_isite_link[e]) == [link_dim_map[e]] for e in edges(is_isite_link))
  @test all(e -> dim.(is_isite_link[e]) == [link_dim_map[e]], keys(link_dim_map))
  is_isite_mlink = IndsNetwork(c; site_space=site_inds_map, link_space=link_dim_map_multi)
  @test is_isite_mlink isa IndsNetwork
  @test all(is_isite_mlink[v] == [site_inds_map[v]] for v in vertices(is_isite_mlink))
  @test all(dim.(is_isite_mlink[e]) == link_dim_map_multi[e] for e in edges(is_isite_mlink))
  @test all(e -> dim.(is_isite_mlink[e]) == link_dim_map_multi[e], keys(link_dim_map_multi))
  # index site dict, index link dict
  is_misite_ilink = IndsNetwork(c; site_space=site_inds_map_multi, link_space=link_inds_map)
  @test is_misite_ilink isa IndsNetwork
  @test all(is_misite_ilink[v] == site_inds_map_multi[v] for v in vertices(is_misite_ilink))
  @test all(is_misite_ilink[e] == [link_inds_map[e]] for e in edges(is_misite_ilink))
  @test all(e -> is_misite_ilink[e] == [link_inds_map[e]], keys(link_inds_map))
  is_misite_milink = IndsNetwork(
    c; site_space=site_inds_map_multi, link_space=link_inds_map_multi
  )
  @test is_misite_ilink isa IndsNetwork
  @test all(
    is_misite_milink[v] == site_inds_map_multi[v] for v in vertices(is_misite_milink)
  )
  @test all(is_misite_milink[e] == link_inds_map_multi[e] for e in edges(is_misite_milink))
  @test all(e -> is_misite_milink[e] == link_inds_map_multi[e], keys(link_inds_map_multi))
end

@testset "IndsNetwork merging" begin
  # test on comb tree
  dims = (3, 2)
  c = named_comb_tree(dims)
  rng = StableRNG(1234)
  site_dims1 = Dictionary(
    vertices(c), [[rand(rng, 2:6) for ni in 1:rand(rng, 1:3)] for _ in 1:nv(c)]
  )
  site_dims2 = Dictionary(
    vertices(c), [[rand(rng, 2:6) for ni in 1:rand(rng, 1:3)] for _ in 1:nv(c)]
  )
  link_dims1 = Dictionary(
    edges(c), [[rand(rng, 2:6) for ni in 1:rand(rng, 1:3)] for _ in 1:ne(c)]
  )
  link_dims2 = Dictionary(
    edges(c), [[rand(rng, 2:6) for ni in 1:rand(rng, 1:3)] for _ in 1:ne(c)]
  )
  is1_s = IndsNetwork(c; site_space=site_dims1)
  is2_s = IndsNetwork(c; site_space=site_dims2)
  is1_e = IndsNetwork(c; link_space=link_dims1)
  is2_e = IndsNetwork(c; link_space=link_dims2)
  is1 = IndsNetwork(c; site_space=site_dims1, link_space=link_dims1)
  is2 = IndsNetwork(c; site_space=site_dims2, link_space=link_dims2)
  # merge some networks
  is1_m = union_all_inds(is1_s, is1_e)
  @test dim.(vertex_data(is1_m)) == dim.(vertex_data(is1))
  is_ms = union_all_inds(is1_s, is2_s)
  @test all(issetequal(is_ms[v], union(is1_s[v], is2_s[v])) for v in vertices(c))
  @test isempty(edge_data(is_ms))
  is_me = union_all_inds(is1_e, is2_e)
  @test all(issetequal(is_me[e], union(is1_e[e], is2_e[e])) for e in edges(c))
  @test isempty(vertex_data(is_me))
  is_m = union_all_inds(is1, is2)
  @test all(issetequal(is_m[v], union(is1[v], is2[v])) for v in vertices(c))
  @test all(issetequal(is_m[e], union(is1[e], is2[e])) for e in edges(c))
end
end
