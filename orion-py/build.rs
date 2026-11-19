/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Adds the platform-specific linker settings required by the PyO3 extension module.
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
