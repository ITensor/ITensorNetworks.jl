function hypercubic_lattice_graph(dims::Tuple{Vararg{Int}})
  return set_vertices(grid(dims), dims)
end
hypercubic_lattice_graph(dims::Int...) = hypercubic_lattice_graph(dims)
square_lattice_graph(dims::Tuple{Int,Int}) = hypercubic_lattice_graph(dims)
square_lattice_graph(dim1::Int, dim2::Int) = hypercubic_lattice_graph((dim1, dim2))
cubic_lattice_graph(dims::Tuple{Int,Int,Int}) = hypercubic_lattice_graph(dims)
cubic_lattice_graph(dim1::Int, dim2::Int, dim3::Int) = hypercubic_lattice_graph((dim1, dim2, dim3))

chain_lattice_graph(dims::Tuple{Int}) = hypercubic_lattice_graph(dims)
# This case is special, and the vertices are just integers
chain_lattice_graph(dim1::Int) = grid((dim1,))
