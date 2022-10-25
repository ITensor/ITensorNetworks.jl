using ITensors
include("../networks/lattices.jl")

struct LocalMPO
  mpo::MPO
  coord1::Tuple{<:Integer,<:Integer}
  coord2::Tuple{<:Integer,<:Integer}
end

struct LineMPO
  mpo::MPO
  coord::Union{Tuple{Colon,<:Integer},Tuple{<:Integer,Colon}}
end

# Transverse field
# The critical point is h = 1.0
# This is the most challenging part of the model for DMRG
function mpo(::Model"tfim", sites::Matrix{<:Index}; h::Float64)
  Ny, Nx = size(sites)
  sites_vec = reshape(sites, Nx * Ny)
  lattice = square_lattice(Nx, Ny; yperiodic=false)

  opsum = OpSum()
  for b in lattice
    opsum += -1, "X", b.s1, "X", b.s2
  end
  for i in 1:(Nx * Ny)
    opsum += h, "Z", i
  end
  return MPO(opsum, sites_vec)
end

function localham_term(
  ::Model"tfim",
  sites::Matrix{<:Index},
  bond::Tuple{Tuple{<:Integer,<:Integer},Tuple{<:Integer,<:Integer}};
  h::Float64,
)
  Ny, Nx = size(sites)
  coord1, coord2 = bond
  opsum = OpSum()
  opsum += -1, "X", 1, "X", 2
  if coord2[1] == coord1[1] + 1
    opsum += h, "Z", 1
  end
  if coord2[1] == coord1[1] + 1 && coord2[1] == Ny
    opsum += h, "Z", 2
  end
  mpo = MPO(opsum, [sites[coord1...], sites[coord2...]])
  return LocalMPO(mpo, coord1, coord2)
end

function localham(m::Model, sites; kwargs...)
  Ny, Nx = size(sites)
  lattice = Square((Ny, Nx))
  bds = bonds(lattice; periodic=false)
  return [localham_term(m, sites, bond; kwargs...) for bond in bds]
end

function localham_term(
  ::Model"tfim", sites::Matrix{<:Index}, bond::Tuple{Colon,<:Integer}; h::Float64
)
  Ny, Nx = size(sites)
  opsum = OpSum()
  for i in 1:(Ny - 1)
    opsum += -1, "X", i, "X", i + 1
    opsum += h, "Z", i
  end
  opsum += h, "Z", Ny
  return LineMPO(MPO(opsum, sites[bond...]), bond)
end

function localham_term(
  ::Model"tfim", sites::Matrix{<:Index}, bond::Tuple{<:Integer,Colon}; h::Float64
)
  Ny, Nx = size(sites)
  opsum = OpSum()
  for i in 1:(Nx - 1)
    opsum += -1, "X", i, "X", i + 1
  end
  return LineMPO(MPO(opsum, sites[bond...]), bond)
end

function lineham(m::Model, sites; kwargs...)
  Ny, Nx = size(sites)
  lattice = Square((Ny, Nx))
  bonds_row = [(i, :) for i in 1:Ny]
  bonds_column = [(:, i) for i in 1:Nx]
  bds = vcat(bonds_row, bonds_column)
  return [localham_term(m, sites, bond; kwargs...) for bond in bds]
end

# Check that the local Hamiltonian is the same as the MPO
function checkham(Hlocal::Array{LocalMPO}, H, sites)
  @disable_warn_order begin
    Ny, Nx = size(sites)
    lattice = Square((Ny, Nx))
    bds = bonds(lattice; periodic=false)
    Hlocal_full = ITensor()
    for (i, bond) in enumerate(bds)
      Hlocalterm_full = prod(Hlocal[i].mpo)
      for y in 1:Ny
        for x in 1:Nx
          if !((y, x) in bond)
            Hlocalterm_full *= op("Id", vec(sites), (x - 1) * Ny + y)
          end
        end
      end
      Hlocal_full += Hlocalterm_full
    end
    @show norm(Hlocal_full - prod(H))
  end
  return isapprox(norm(Hlocal_full), norm(prod(H)))
end

function checkham(Hline::Array{LineMPO}, H, sites)
  @disable_warn_order begin
    Ny, Nx = size(sites)
    Hlocal_full = ITensor()
    for h in Hline
      h_full = prod(h.mpo)
      if h.coord[1] isa Colon
        for y in 1:Ny
          for x in 1:Nx
            if x != h.coord[2]
              h_full *= op("Id", vec(sites), (x - 1) * Ny + y)
            end
          end
        end
      else
        for y in 1:Ny
          for x in 1:Nx
            if y != h.coord[1]
              h_full *= op("Id", vec(sites), (x - 1) * Ny + y)
            end
          end
        end
      end
      Hlocal_full += h_full
    end
    @show norm(Hlocal_full - prod(H))
  end
  return isapprox(norm(Hlocal_full), norm(prod(H)))
end
