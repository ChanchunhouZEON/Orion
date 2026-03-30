/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::common::{ANNError, ANNResult};

const SPACE_FOR_CACHED_NODES_IN_GB: f64 = 0.25;
const THRESHOLD_FOR_CACHING_IN_GB: f64 = 1.0;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DiskIndexBuildParameters {
    search_ram_limit: f64,
    index_build_ram_limit: f64,
}

impl DiskIndexBuildParameters {
    pub fn new(search_ram_limit_gb: f64, index_build_ram_limit_gb: f64) -> ANNResult<Self> {
        let param = Self {
            search_ram_limit: Self::get_memory_budget(search_ram_limit_gb),
            index_build_ram_limit: index_build_ram_limit_gb * 1024_f64 * 1024_f64 * 1024_f64,
        };

        if param.search_ram_limit <= 0f64 {
            return Err(ANNError::log_index_config_error(
                "search_ram_limit".to_string(),
                "RAM budget should be > 0".to_string(),
            ));
        }

        if param.index_build_ram_limit <= 0f64 {
            return Err(ANNError::log_index_config_error(
                "index_build_ram_limit".to_string(),
                "RAM budget should be > 0".to_string(),
            ));
        }

        Ok(param)
    }

    pub fn search_ram_limit(&self) -> f64 {
        self.search_ram_limit
    }

    pub fn index_build_ram_limit(&self) -> f64 {
        self.index_build_ram_limit
    }

    fn get_memory_budget(mut index_ram_limit_gb: f64) -> f64 {
        if index_ram_limit_gb - SPACE_FOR_CACHED_NODES_IN_GB > THRESHOLD_FOR_CACHING_IN_GB {
            index_ram_limit_gb -= SPACE_FOR_CACHED_NODES_IN_GB;
        }
        index_ram_limit_gb * 1024_f64 * 1024_f64 * 1024_f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sufficient_ram_for_caching() {
        let param = DiskIndexBuildParameters::new(1.26_f64, 1.0_f64).unwrap();
        assert_eq!(
            param.search_ram_limit,
            1.01_f64 * 1024_f64 * 1024_f64 * 1024_f64
        );
    }

    #[test]
    fn insufficient_ram_for_caching() {
        let param = DiskIndexBuildParameters::new(0.03_f64, 1.0_f64).unwrap();
        assert_eq!(
            param.search_ram_limit,
            0.03_f64 * 1024_f64 * 1024_f64 * 1024_f64
        );
    }
}
