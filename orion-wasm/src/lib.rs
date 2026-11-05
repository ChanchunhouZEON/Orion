/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Browser-facing replay of native Orion traces. No graph search or filesystem
//! support is implied by this initial interface. Run in a Web Worker for large traces.
// Keep bindings available to host type checking and rust-analyzer. JavaScript
// interop is exercised by the actual WASM replay test, not by native execution.
mod bindings {
    use orion_ann_diagnostics::{Error, TraceReplay};
    use serde::Serialize;
    use wasm_bindgen::prelude::*;

    fn report(error: Error) -> JsValue {
        error
            .report()
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or_else(|_| JsValue::from_str("failed to serialize Orion error report"))
    }

    /// Owns a validated, bounded trace. JavaScript calls `free()` to release it.
    #[wasm_bindgen(js_name = TraceReplay)]
    pub struct WasmTraceReplay {
        replay: TraceReplay,
    }

    #[wasm_bindgen(js_class = TraceReplay)]
    impl WasmTraceReplay {
        #[wasm_bindgen(constructor)]
        pub fn new(json: &str) -> Result<WasmTraceReplay, JsValue> {
            Ok(Self {
                replay: TraceReplay::from_json(json).map_err(report)?,
            })
        }
        #[wasm_bindgen(getter)]
        pub fn position(&self) -> u32 {
            self.replay.position() as u32
        }
        #[wasm_bindgen(getter, js_name = eventCount)]
        pub fn event_count(&self) -> u32 {
            self.replay.trace().events.len() as u32
        }
        #[wasm_bindgen(getter)]
        pub fn truncated(&self) -> bool {
            self.replay.trace().truncated
        }
        pub fn seek(&mut self, position: u32) -> Result<(), JsValue> {
            self.replay.seek(position as usize).map_err(report)
        }
        /// Returns an ordinary JavaScript array; no borrowed WASM heap views escape.
        #[wasm_bindgen(js_name = nextEvents)]
        pub fn next_events(&mut self, count: u32) -> Result<JsValue, JsValue> {
            let events = self.replay.next_events(count as usize).map_err(report)?;
            events
                .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
                .map_err(|e| report(Error::invalid_data(e.to_string()).at("trace")))
        }
        #[wasm_bindgen(js_name = resultIds)]
        pub fn result_ids(&self) -> Vec<u32> {
            self.replay.trace().result_ids.clone()
        }
        pub fn metadata(&self) -> Result<JsValue, JsValue> {
            serde_wasm_bindgen::to_value(&self.replay.trace().metadata)
                .map_err(|e| report(Error::invalid_data(e.to_string()).at("trace")))
        }
    }
}
