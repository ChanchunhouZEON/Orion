// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
import type { Catalog, Dataset, Section } from "../types";
import {
  defaultCascadeIds,
  defaultHiddenEngines,
  supports,
} from "../model/selection.mjs";

export const sections: { id: Section; label: string }[] = [
  { id: "default", label: "Default" },
  { id: "cascades", label: "Cascades" },
  { id: "ablation", label: "Ablation" },
  { id: "build", label: "Build" },
  { id: "memory", label: "Memory" },
  { id: "threads", label: "Thread scaling" },
];

/** Shared navigation/session state. Catalog views own their derived measurements.
 * Keep this instance across view switches to preserve the existing selections.
 */
export class AtlasState {
  readonly catalog: Catalog;
  readonly datasets: Dataset[];
  datasetId = $state("sift");
  section = $state<Section>("default");
  hidden = $state<string[]>([]);
  detail = $state("");
  target = $state(0.95);
  logarithmic = $state(true);
  cascadeIds = $state<string[]>([]);
  cascadeMinimum = $state(0.9);
  allCascadePoints = $state(false);
  studyId = $state("");
  armId = $state("");
  compare = $state(false);
  ablationView = $state<"bars" | "curves">("bars");
  speedup = $state(false);

  constructor(catalog: Catalog, datasets: Dataset[]) {
    this.catalog = catalog;
    this.datasets = datasets;
    this.selectDataset("sift");
  }
  get dataset(): Dataset {
    return this.datasets.find((d) => d.id === this.datasetId)!;
  }
  get study() {
    return (
      this.dataset.ablations.find((a) => a.id === this.studyId) ||
      this.dataset.ablations[0]
    );
  }
  selectDataset(id: string) {
    const next = this.datasets.find((d) => d.id === id)!;
    this.datasetId = id;
    this.cascadeIds = defaultCascadeIds(next);
    this.studyId = next.ablations[0]?.id || "";
    this.armId = next.ablations[0]?.variants[0]?.id || "";
    this.compare = false;
    this.detail = "";
    this.ablationView = "bars";
    this.hidden = defaultHiddenEngines(next, this.section);
  }
  selectSection(id: Section) {
    this.section = id;
    this.compare = false;
    this.detail = "";
    this.ablationView = "bars";
    this.hidden = defaultHiddenEngines(this.dataset, id);
    if (!supports(this.dataset, id))
      this.selectDataset(this.datasets.find((d) => supports(d, id))!.id);
  }
  updateCascades(ids: string[]) {
    this.cascadeIds = ids;
    this.hidden = this.hidden.filter((e) => e !== "orion");
    this.detail = "";
  }
  selectStudy(id: string) {
    this.studyId = id;
    this.armId = this.dataset.ablations.find(
      (a) => a.id === id,
    )!.defaultVariant;
    this.compare = false;
    this.ablationView = "bars";
    this.detail = "";
    this.hidden = [];
  }
  toggleEngine(engine: string) {
    this.hidden = this.hidden.includes(engine)
      ? this.hidden.filter((e) => e !== engine)
      : [...this.hidden, engine];
  }
}
