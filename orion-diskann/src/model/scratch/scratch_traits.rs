/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Resets reusable query state before returning it to the shared scratch pool.
pub trait Scratch {
    /// Resets query-specific state while retaining reusable allocation capacity before returning scratch to the pool.
    fn clear(&mut self);
}
