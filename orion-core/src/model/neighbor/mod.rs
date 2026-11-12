/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#[expect(
    clippy::module_inception,
    reason = "preserve the existing neighbor type module path"
)]
pub mod neighbor;
pub use neighbor::*;

pub mod neighbor_priority_queue;
pub use neighbor_priority_queue::*;
