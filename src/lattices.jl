function hypercubic_lattice_graph(dims::Tuple{Vararg{Int}})
  return NamedGraph(grid(dims); vertices=dims)
end
function hypercubic_lattice_graph(dim::Int)
  return NamedGraph(grid((dim,)), 1:dim)
end
function hypercubic_lattice_graph(dim1::Int, dim2::Int, dims::Int...)
  return hypercubic_lattice_graph((dim1, dim2, dims...))
end
square_lattice_graph(dims::Tuple{Int,Int}) = hypercubic_lattice_graph(dims)
square_lattice_graph(dim1::Int, dim2::Int) = hypercubic_lattice_graph((dim1, dim2))
cubic_lattice_graph(dims::Tuple{Int,Int,Int}) = hypercubic_lattice_graph(dims)
function cubic_lattice_graph(dim1::Int, dim2::Int, dim3::Int)
  return hypercubic_lattice_graph((dim1, dim2, dim3))
end

chain_lattice_graph(dims::Tuple{Int}) = hypercubic_lattice_graph(dims)
# This case is special, and the vertices are just integers
chain_lattice_graph(dim::Int) = hypercubic_lattice_graph(dim)
