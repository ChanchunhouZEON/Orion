// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
import type { Catalog, Dataset } from "../types";
/** Fetch only the self-contained public catalog. Injectable for offline tests. */
export async function loadAtlas(
  baseUrl: string,
  request: typeof fetch = fetch,
): Promise<{ catalog: Catalog; datasets: Dataset[] }> {
  const load = async (name: string) => {
    const response = await request(`${baseUrl}data/${name}.json`);
    if (!response.ok) throw new Error(`Cannot load ${name}`);
    return response.json();
  };
  const catalog: Catalog = await load("catalog");
  const datasets: Dataset[] = await Promise.all(
    catalog.datasets.map((d) => load(d.id)),
  );
  return { catalog, datasets };
}
