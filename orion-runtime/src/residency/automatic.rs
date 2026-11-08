/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Deployment planning only: no cascade selection, allocations or search work.
use super::*;
use orion_ann_platform::{residency::registered_bytes, resources::MemoryResources};

#[derive(Debug)]
pub(crate) struct AutomaticPlan {
    pub options: PinOptions,
    pub observations: serde_json::Value,
}

impl AutomaticPlan {
    pub fn detect(enabled: bool, budget: Option<f64>) -> Result<Self> {
        Self::from_resources(
            enabled,
            budget,
            MemoryResources::detect(),
            registered_bytes(),
        )
    }

    fn from_resources(
        enabled: bool,
        budget: Option<f64>,
        resources: MemoryResources,
        registered: u64,
    ) -> Result<Self> {
        crate::config::resources::validate_memory_budget(budget)?;
        // Leave at least 256 MiB and 10% of deployment memory for other users,
        // scratch and runtime overhead. Available memory already excludes the
        // resident index: subtracting its RSS again would double count it.
        let reserve = resources
            .total_bytes
            .map_or(256 << 20, |n| (n / 10).max(256 << 20));
        let headroom = resources
            .available_bytes
            .unwrap_or(0)
            .saturating_sub(reserve);
        let locked = resources.locked_bytes.unwrap_or(0).max(registered);
        let lock_headroom = resources
            .lock_limit_bytes
            .unwrap_or(0)
            .saturating_sub(locked);
        let user = budget.map_or(u64::MAX, |gib| {
            crate::config::resources::gib_to_bytes(gib) as u64
        });
        let bytes = if enabled {
            headroom.min(lock_headroom).min(user)
        } else {
            0
        };
        let budget_bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
        Ok(Self {
            options: PinOptions {
                mode: if enabled {
                    PinMode::BestEffort
                } else {
                    PinMode::Off
                },
                budget_bytes,
                datasets: None,
            },
            observations: serde_json::json!({
                "enabled": enabled, "budget_bytes": budget_bytes,
                "total_bytes": resources.total_bytes, "available_bytes": resources.available_bytes,
                "reserve_bytes": reserve, "lock_limit_bytes": resources.lock_limit_bytes,
                "observed_locked_bytes": resources.locked_bytes, "registered_locked_bytes": registered,
                "policy": "best-effort; unknown headroom or lock allowance disables automatic locks",
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn resources() -> MemoryResources {
        MemoryResources {
            total_bytes: Some(16 << 30),
            available_bytes: Some(8 << 30),
            lock_limit_bytes: Some(4 << 30),
            locked_bytes: Some(1 << 30),
        }
    }
    #[test]
    fn independent_user_headroom_and_lock_limits() {
        assert_eq!(
            AutomaticPlan::from_resources(true, None, resources(), 0)
                .unwrap()
                .options
                .budget_bytes,
            3 << 30
        );
        assert_eq!(
            AutomaticPlan::from_resources(true, Some(1.), resources(), 0)
                .unwrap()
                .options
                .budget_bytes,
            1 << 30
        );
        assert_eq!(
            AutomaticPlan::from_resources(true, None, resources(), 3 << 30)
                .unwrap()
                .options
                .budget_bytes,
            1 << 30
        );
        let mut r = resources();
        r.available_bytes = Some(1 << 30);
        assert_eq!(
            AutomaticPlan::from_resources(true, None, r, 0)
                .unwrap()
                .options
                .budget_bytes,
            0
        );
    }
    #[test]
    fn unknown_exhausted_and_disabled_never_lock_unbounded_memory() {
        for r in [
            MemoryResources::default(),
            MemoryResources {
                lock_limit_bytes: None,
                ..resources()
            },
            MemoryResources {
                available_bytes: None,
                ..resources()
            },
        ] {
            assert_eq!(
                AutomaticPlan::from_resources(true, None, r, 0)
                    .unwrap()
                    .options
                    .budget_bytes,
                0
            );
        }
        let off = AutomaticPlan::from_resources(false, None, resources(), 0).unwrap();
        assert_eq!(off.options.mode, PinMode::Off);
        assert_eq!(off.options.budget_bytes, 0);
        assert!(AutomaticPlan::from_resources(true, Some(f64::NAN), resources(), 0).is_err());
        assert!(AutomaticPlan::from_resources(true, Some(-1.), resources(), 0).is_err());
    }
}
