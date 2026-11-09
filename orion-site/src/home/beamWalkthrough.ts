// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.

// Shared illustrative state for the graph and beam. Positions are not a metric embedding.
export const beamNodes = [
  { id: "A", distance: 0.9, x: 55, y: 120, labelY: 148 },
  { id: "B", distance: 0.6, x: 170, y: 160, labelY: 186 },
  { id: "C", distance: 0.2, x: 295, y: 45, labelY: 23 },
  { id: "D", distance: 0.4, x: 290, y: 140, labelY: 168 },
];
export const beamEdges = [
  ["A", "B"],
  ["A", "C"],
  ["C", "D"],
  ["D", "B"],
];
export interface BeamFrame {
  title: string;
  text: string;
  beam: string[];
  expanded: string[];
  pending?: string;
  added?: string;
  result?: string[];
}
export const beamFrames: BeamFrame[] = [
  {
    title: "Start at an entry point",
    text: "Score entry A at 0.90. The beam has one candidate and two free slots.",
    beam: ["A"],
    expanded: [],
  },
  {
    title: "Expand A and fill the beam",
    text: "Expand A and score unseen neighbors B (0.60) and C (0.20). Both enter the beam. A is marked expanded but remains a candidate.",
    beam: ["C", "B", "A"],
    expanded: ["A"],
  },
  {
    title: "Expand C and score D",
    text: "C is the closest unexpanded candidate. Its neighbor D scores 0.40, better than the current worst entry A at 0.90. D is pending admission.",
    beam: ["C", "B", "A"],
    expanded: ["A", "C"],
    pending: "D",
  },
  {
    title: "Admit D and replace A",
    text: "D enters the full beam and displaces A. C remains the best candidate even though it has already been expanded.",
    beam: ["C", "D", "B"],
    expanded: ["A", "C"],
    added: "D",
  },
  {
    title: "Finish expansion and return top-k",
    text: "Expand D, then B. D leads to B, which is already visited, and B has no outgoing edges here. No unexpanded beam candidates remain. Return C and D: k=2 results from the L=3 pool.",
    beam: ["C", "D", "B"],
    expanded: ["A", "C", "D", "B"],
    result: ["C", "D"],
  },
];
