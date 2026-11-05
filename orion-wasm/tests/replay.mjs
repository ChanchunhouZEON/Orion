/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

// Runs the browser-target JS glue and actual WASM in Node's WebAssembly VM.
// No search implementation or mocked binding is substituted here.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [directory, fixture] = process.argv.slice(2);
const pkg = resolve(directory);
const api = await import(pathToFileURL(`${pkg}/orion_ann_wasm.js`).href);
api.initSync({ module: readFileSync(`${pkg}/orion_ann_wasm_bg.wasm`) });
const read = (path) => JSON.parse(readFileSync(path, "utf8"));
const expected = read(fixture);
const replay = new api.TraceReplay(JSON.stringify(expected));
assert.equal(replay.position, 0);
assert.equal(replay.eventCount, expected.events.length);
assert.equal(replay.truncated, false);
assert.deepEqual(replay.metadata(), expected.metadata);
assert.deepEqual(Array.from(replay.resultIds()), expected.result_ids);
let events = [];
while (replay.position < replay.eventCount)
  events.push(...replay.nextEvents(7));
assert.deepEqual(events, expected.events);
assert.deepEqual(replay.nextEvents(1), []);
replay.seek(0);
assert.deepEqual(replay.nextEvents(1), expected.events.slice(0, 1));
const before = replay.position;
for (const action of [
  () => replay.seek(0xffffffff),
  () => replay.nextEvents(0),
  () => replay.nextEvents(1025),
]) {
  assert.throws(
    action,
    (e) => e.code === "ORION_INVALID_ARGUMENT" && e.operation === "trace",
  );
  assert.equal(replay.position, before);
}
const bad = { ...expected, schema_version: 999 };
assert.throws(
  () => new api.TraceReplay(JSON.stringify(bad)),
  (error) => {
    assert.deepEqual(error, read(fixture.replace(/\.json$/, ".error.json")));
    return true;
  },
);
assert.throws(
  () => new api.TraceReplay("{}"),
  (e) => e.code === "ORION_INVALID_DATA",
);
assert.throws(
  () => new api.TraceReplay(" ".repeat(8 * 1024 * 1024 + 1)),
  (e) => e.code === "ORION_INVALID_DATA",
);
const prefix = new api.TraceReplay(
  readFileSync(fixture.replace(/\.json$/, ".truncated.json"), "utf8"),
);
assert.equal(prefix.truncated, true);
assert.equal(prefix.eventCount, 1);
assert.deepEqual(Array.from(prefix.resultIds()), expected.result_ids);
prefix.free();
replay.free();
console.log(
  JSON.stringify({
    status: "passed",
    actual_wasm: true,
    events: events.length,
    native_trace_parity: true,
    native_error_parity: true,
    truncated_results_preserved: true,
  }),
);
