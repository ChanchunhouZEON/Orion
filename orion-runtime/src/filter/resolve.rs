/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Produce sorted unique row IDs from a single immutable label view.
use super::LabelFilter;
use crate::{
    labels::{update::union, LabelSnapshot},
    Error, Result,
};

pub(crate) fn resolve(
    snapshot: &LabelSnapshot,
    filter: &LabelFilter,
    limit: usize,
) -> Result<Vec<u32>> {
    if filter
        .all
        .len()
        .saturating_add(filter.any.len())
        .saturating_add(filter.none.len())
        > 4096
    {
        return Err(Error::invalid_argument("filter exceeds 4096 label terms"));
    }
    for key in filter.all.iter().chain(&filter.any).chain(&filter.none) {
        key.validate()?;
    }
    let get = |key| {
        snapshot
            .labels
            .get(key)
            .map_or(&[][..], |ids| ids.as_slice())
    };
    let any = filter
        .any
        .iter()
        .fold(Vec::new(), |acc, key| union(&acc, get(key)));
    let denied = filter
        .none
        .iter()
        .fold(Vec::new(), |acc, key| union(&acc, get(key)));
    let all_seed = filter.all.iter().map(get).min_by_key(|ids| ids.len());
    let seed = all_seed.or_else(|| (!filter.any.is_empty()).then_some(any.as_slice()));
    if let Some(seed) = seed {
        let mut ids = Vec::new();
        for &id in seed {
            if filter
                .all
                .iter()
                .all(|key| get(key).binary_search(&id).is_ok())
                && (filter.any.is_empty() || any.binary_search(&id).is_ok())
                && denied.binary_search(&id).is_err()
            {
                if ids.len() == limit {
                    return Err(limit_error(limit));
                }
                ids.push(id);
            }
        }
        Ok(ids)
    } else {
        // Complement cardinality is known before scanning the universe.
        if snapshot.points - denied.len() > limit {
            return Err(limit_error(limit));
        }
        // Emit gaps, so a mostly excluded universe does not require N lookups.
        let mut ids = Vec::with_capacity(snapshot.points - denied.len());
        let mut start = 0u32;
        for &id in &denied {
            ids.extend(start..id);
            start = id + 1;
        }
        ids.extend(start..snapshot.points as u32);
        Ok(ids)
    }
}
fn limit_error(limit: usize) -> Error {
    Error::invalid_argument("filtered exact scan exceeds max_candidates")
        .with_detail("max_candidates", limit)
}
