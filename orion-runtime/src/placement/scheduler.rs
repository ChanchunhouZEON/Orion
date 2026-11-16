/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Preparation-time placement using expected access frequency, never hot-loop
//! counters. This is a deterministic hint policy, not online eviction or RSS control.
use super::{
    object::{ObjectDecision, ObjectDescriptor},
    StorageMode, StorageObject, StorageOptions,
};
use crate::{Error, Result};
use orion_ann_platform::region::Placement;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Frequency {
    Cold,
    Rerank,
    Prefilter,
    Admission,
    Graph,
}

pub(super) struct Selection {
    pub objects: Vec<ObjectDecision>,
    pub inactive_overrides: Vec<StorageObject>,
    pub placements: BTreeMap<&'static str, Placement>,
    pub resident: u64,
    pub mapped: u64,
}

/// Reserve graph/admission before colder objects, regardless of allocation order.
/// If a hot object does not fit, let smaller objects use remaining headroom.
pub(super) fn select(
    mut objects: Vec<ObjectDescriptor>,
    limit: Option<u64>,
    fixed: u64,
    policy: &StorageOptions,
) -> Result<Selection> {
    // Only resident payloads consume this budget. Mapped bytes are tracked
    // separately; the OS controls their page-cache residency, not this planner.
    // Keep reporting possible when fixed bytes already exceed the budget;
    // the caller's enforce() rejects that case.
    // Reserve all mandatory resident bytes before automatic candidates. Reports
    // remain available even when over budget; execution calls enforce afterwards.
    let mandatory = objects
        .iter()
        .filter(|d| d.mode == StorageMode::Resident)
        .try_fold(fixed, |n, d| n.checked_add(d.bytes))
        .ok_or_else(|| Error::invalid_argument("mandatory storage size overflow"))?;
    let mut available = limit.unwrap_or(u64::MAX).saturating_sub(mandatory);
    let mut placements = BTreeMap::new();
    let mut decisions = Vec::new();
    objects.sort_by(|a, b| b.frequency.cmp(&a.frequency).then(a.name.cmp(b.name)));
    for object in &objects {
        if placements.contains_key(object.name) {
            return Err(Error::invalid_argument("duplicate placement object"));
        }
        let placement = match object.mode {
            StorageMode::Mapped => Placement::Mapped,
            StorageMode::Resident => Placement::Resident,
            StorageMode::Auto if object.bytes <= available => {
                available -= object.bytes;
                Placement::Resident
            }
            StorageMode::Auto => Placement::Mapped,
        };
        decisions.push(ObjectDecision::new(
            object,
            placement,
            policy.objects.contains_key(&object.object),
        ));
        placements.insert(object.name, placement);
    }
    let inactive_overrides = policy
        .objects
        .keys()
        .filter(|key| !objects.iter().any(|d| d.object == **key))
        .copied()
        .collect();
    let mut result = Selection {
        objects: decisions,
        inactive_overrides,
        placements,
        resident: 0,
        mapped: 0,
    };
    for object in objects {
        let total = if result.placements[object.name] == Placement::Resident {
            &mut result.resident
        } else {
            &mut result.mapped
        };
        *total = total
            .checked_add(object.bytes)
            .ok_or_else(|| Error::invalid_argument("placement size overflow"))?;
    }
    // This is a forecast only. Actual buffers acquire their own allocation leases.
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn select(
        mut objects: Vec<ObjectDescriptor>,
        mode: StorageMode,
        limit: Option<u64>,
        fixed: u64,
    ) -> Result<Selection> {
        for object in &mut objects {
            object.mode = mode;
        }
        super::select(
            objects,
            limit,
            fixed,
            &StorageOptions {
                mode,
                ..Default::default()
            },
        )
    }
    const M: u64 = 1024 * 1024;
    fn demand(name: &'static str, bytes: u64, frequency: Frequency) -> ObjectDescriptor {
        ObjectDescriptor {
            object: StorageObject::from_codec(name).unwrap(),
            mode: StorageMode::Auto,
            name,
            bytes,
            frequency,
        }
    }
    #[test]
    fn keeps_graph_and_admission_even_when_cold_base_is_allocated_first() {
        let s = select(
            vec![
                demand("base", 100 * M, Frequency::Cold),
                demand("graph", 80 * M, Frequency::Graph),
                demand("qds", 30 * M, Frequency::Admission),
            ],
            StorageMode::Auto,
            Some(110 * M),
            0,
        )
        .unwrap();
        assert_eq!(s.placements["base"], Placement::Mapped);
        assert_eq!(s.placements["graph"], Placement::Resident);
        assert_eq!(s.placements["qds"], Placement::Resident);
        // A cold mapping must not displace hot payloads that fit exactly.
        assert_eq!((s.resident, s.mapped), (110 * M, 100 * M));
    }
    #[test]
    fn skips_oversized_hot_objects_without_wasting_remaining_capacity() {
        let s = select(
            vec![
                demand("graph", 200 * M, Frequency::Graph),
                demand("qds", 10 * M, Frequency::Admission),
            ],
            StorageMode::Auto,
            Some(80 * M),
            0,
        )
        .unwrap();
        assert_eq!(s.placements["graph"], Placement::Mapped);
        assert_eq!(s.placements["qds"], Placement::Resident);
    }
    #[test]
    fn resident_payload_can_fill_the_budget_exactly() {
        let s = select(
            vec![demand("base", 10, Frequency::Cold)],
            StorageMode::Auto,
            Some(10),
            0,
        )
        .unwrap();
        assert_eq!((s.resident, s.mapped), (10, 0));
    }

    #[test]
    fn mapped_objects_need_no_resident_budget() {
        for mode in [StorageMode::Auto, StorageMode::Mapped] {
            let s = select(
                vec![demand("graph", 200 * M, Frequency::Graph)],
                mode,
                Some(0),
                0,
            )
            .unwrap();
            assert_eq!(s.placements["graph"], Placement::Mapped);
            assert_eq!((s.resident, s.mapped), (0, 200 * M));
        }
    }
    #[test]
    fn partial_overrides_reserve_mandatory_objects_before_priority_and_ignore_input_order() {
        let policy = StorageOptions {
            mode: StorageMode::Auto,
            objects: [
                (StorageObject::Base, StorageMode::Resident),
                (StorageObject::L2U8, StorageMode::Mapped),
            ]
            .into(),
            ..Default::default()
        };
        let run = |reverse| {
            let mut objects = vec![
                demand("base", 40, Frequency::Cold),
                demand("graph", 50, Frequency::Graph),
                demand("qds", 5, Frequency::Admission),
                demand("jls", 10, Frequency::Prefilter),
                demand("qds6", 20, Frequency::Rerank),
            ];
            for object in &mut objects {
                object.mode = policy.requested(object.object);
            }
            if reverse {
                objects.reverse();
            }
            super::select(objects, Some(100), 0, &policy).unwrap()
        };
        let result = run(false);
        assert_eq!(result.placements, run(true).placements);
        assert_eq!((result.resident, result.mapped), (100, 25));
        assert_eq!(result.placements["base"], Placement::Resident);
        assert_eq!(result.placements["graph"], Placement::Resident);
        assert_eq!(result.placements["jls"], Placement::Resident);
        assert_eq!(result.placements["qds"], Placement::Mapped);
        assert_eq!(result.placements["qds6"], Placement::Mapped);
    }
}
