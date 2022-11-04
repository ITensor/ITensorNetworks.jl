import Base:
  # types
  Vector,
  # functions
  convert,
  copy,
  getindex,
  hvncat,
  setindex!,
  show,
  isassigned

import .DataGraphs: underlying_graph, vertex_data, edge_data

import Graphs: Graph, is_directed

import LinearAlgebra: svd, factorize, qr, normalize, normalize!

import NamedGraphs: vertex_to_parent_vertex, to_vertex, incident_edges

import ITensors:
  # contraction
  contract,
  contract!,
  orthogonalize,
  orthogonalize!,
  isortho,
  inner,
  loginner,
  norm,
  lognorm,
  # truncation
  truncate!,
  truncate,
  replacebond!,
  replacebond,
  # site and link indices
  siteind,
  siteinds,
  linkinds,
  # index set functions
  uniqueinds,
  commoninds,
  replaceinds,
  hascommoninds,
  # priming and tagging
  adjoint,
  sim,
  prime,
  setprime,
  noprime,
  replaceprime,
  addtags,
  removetags,
  replacetags,
  settags,
  tags,
  # dag
  dag,
  # permute
  permute,
  #commoninds
  check_hascommoninds,
  hascommoninds,
  # linkdims
  linkdim,
  linkdims,
  maxlinkdim,
  # projected operators
  position!,
  set_nsite!,
  product,
  nsite,
  # promotion and conversion
  promote_itensor_eltype,
  scalartype

using ITensors.ContractionSequenceOptimization: deepmap

import ITensors.ITensorVisualizationCore: visualize
