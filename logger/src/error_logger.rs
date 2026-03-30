/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use log::error;

/// Log an error message using the standard log crate.
pub fn log_error(error_message: String) -> Result<(), crate::LogError> {
    error!("{}", error_message);
    Ok(())
}

#[cfg(test)]
mod error_logger_test {
    use super::*;

    #[test]
    fn log_error_works() {
        crate::init_logger();
        log_error(String::from("Test error")).unwrap();
    }
}
