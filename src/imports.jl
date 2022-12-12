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
  isassigned,
  union

import NamedGraphs:
  vertextype,
  convert_vertextype,
  vertex_to_parent_vertex,
  rename_vertices,
  disjoint_union,
  incident_edges

import .DataGraphs:
  underlying_graph,
  underlying_graph_type,
  vertex_data,
  edge_data,
  reverse_data_direction

import Graphs: SimpleGraph, is_directed

import LinearAlgebra: svd, factorize, qr, normalize, normalize!

import ITensors:
  # contraction
  contract,
  orthogonalize,
  orthogonalize!,
  isortho,
  inner,
  loginner,
  norm,
  lognorm,
  expect,
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
