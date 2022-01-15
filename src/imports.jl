import Base:
  convert,
  copy,
  getindex,
  setindex!,
  isassigned

import .DataGraphs: underlying_graph, vertex_data, edge_data

import Graphs: Graph

import ITensors:
  # site and link indices
  siteind,
  siteinds,
  linkinds,
  # index set functions
  uniqueinds,
  commoninds,
  replaceinds,
  # priming and tagging
  sim,
  prime,
  setprime,
  noprime,
  replaceprime,
  addtags,
  removetags,
  replacetags,
  settags,
  # dag
  dag

import ITensors.ITensorVisualizationCore:
  visualize
