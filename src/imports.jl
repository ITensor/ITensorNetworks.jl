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

import .DataGraphs: underlying_graph, underlying_graph_type, vertex_data, edge_data

import Graphs: Graph, is_directed

import LinearAlgebra: svd, factorize, qr

import NamedGraphs:
  vertextype,
  vertex_to_parent_vertex,
  rename_vertices,
  disjoint_union
  #, to_vertex

import ITensors:
  # contraction
  contract,
  orthogonalize,
  inner,
  norm,
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
  dag

using ITensors.ContractionSequenceOptimization: deepmap

import ITensors.ITensorVisualizationCore: visualize
