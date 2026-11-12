/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Serialize writers; readers clone an Arc under a short publication lock.
use super::{update, LabelSnapshot, LabelUpdate, MAX_LABELS, MAX_MEMBERSHIPS};
use crate::{Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, RwLock},
};

pub(crate) struct LabelStore {
    writer: Mutex<()>,
    current: RwLock<Arc<LabelSnapshot>>,
}
impl LabelStore {
    pub fn empty(identity: String, points: usize) -> Self {
        Self::from_snapshot(LabelSnapshot {
            identity,
            points,
            revision: 0,
            labels: BTreeMap::new(),
        })
    }
    pub fn from_snapshot(snapshot: LabelSnapshot) -> Self {
        Self {
            writer: Mutex::new(()),
            current: RwLock::new(Arc::new(snapshot)),
        }
    }
    pub fn snapshot(&self) -> Result<Arc<LabelSnapshot>> {
        self.current
            .read()
            .map(|s| Arc::clone(&s))
            .map_err(|_| Error::invalid_state("label publication lock is poisoned"))
    }
    pub fn update(&self, expected: u64, changes: &[LabelUpdate]) -> Result<u64> {
        let _writer = self
            .writer
            .lock()
            .map_err(|_| Error::invalid_state("label writer is poisoned"))?;
        let previous = self.snapshot()?;
        if previous.revision != expected {
            return Err(Error::invalid_state("label revision conflict")
                .with_detail("expected_revision", expected)
                .with_detail("actual_revision", previous.revision));
        }
        update::check_batch_limits(
            changes.len(),
            changes.iter().map(|c| (c.add.len(), c.remove.len())),
        )?;
        let mut seen = BTreeSet::new();
        let mut labels = previous.labels.clone();
        for change in changes {
            change.label.validate()?;
            if !seen.insert(&change.label) {
                return Err(Error::invalid_argument(
                    "each label may appear only once per update batch",
                ));
            }
            let add = update::checked_ids(&change.add, previous.points)?;
            let remove = update::checked_ids(&change.remove, previous.points)?;
            if add.iter().any(|id| remove.binary_search(id).is_ok()) {
                return Err(Error::invalid_argument("label add/remove sets overlap"));
            }
            let old = labels
                .get(&change.label)
                .map_or(&[][..], |ids| ids.as_slice());
            let next = update::union(&update::difference(old, &remove), &add);
            if next.is_empty() {
                labels.remove(&change.label);
            } else if old != next {
                labels.insert(change.label.clone(), Arc::new(next));
            }
        }
        if labels.len() > MAX_LABELS
            || labels.values().map(|v| v.len()).sum::<usize>() > MAX_MEMBERSHIPS
        {
            return Err(Error::invalid_argument(
                "label state exceeds 4096 labels or 1000000 memberships",
            ));
        }
        if labels == previous.labels {
            return Ok(previous.revision);
        }
        let revision = previous
            .revision
            .checked_add(1)
            .ok_or_else(|| Error::invalid_state("label revision exhausted"))?;
        let next = Arc::new(LabelSnapshot {
            identity: previous.identity.clone(),
            points: previous.points,
            revision,
            labels,
        });
        // Everything fallible is complete before replacing the live generation.
        *self
            .current
            .write()
            .map_err(|_| Error::invalid_state("label publication lock is poisoned"))? = next;
        Ok(revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LabelKey;
    fn update(value: &str, ids: Vec<u32>) -> LabelUpdate {
        LabelUpdate {
            label: LabelKey::new("test", value).unwrap(),
            add: ids,
            remove: vec![],
        }
    }
    #[test]
    fn old_views_survive_publication_and_unchanged_sets_are_shared() {
        let store = LabelStore::empty("test".into(), 10);
        store
            .update(0, &[update("a", vec![1]), update("b", vec![2])])
            .unwrap();
        let old = store.snapshot().unwrap();
        store.update(1, &[update("a", vec![3])]).unwrap();
        let current = store.snapshot().unwrap();
        let a = LabelKey::new("test", "a").unwrap();
        let b = LabelKey::new("test", "b").unwrap();
        assert_eq!(old.labels[&a].as_slice(), [1]);
        assert_eq!(current.labels[&a].as_slice(), [1, 3]);
        assert!(Arc::ptr_eq(&old.labels[&b], &current.labels[&b]));
        assert_eq!(old.revision, 1);
    }
    #[test]
    fn limits_and_revision_exhaustion_do_not_publish() {
        let store = LabelStore::empty("test".into(), 10);
        assert!(store
            .update(0, &[update("a", vec![1; MAX_MEMBERSHIPS + 1])])
            .is_err());
        let too_many: Vec<_> = (0..MAX_LABELS + 1)
            .map(|i| update(&i.to_string(), vec![1]))
            .collect();
        assert!(store.update(0, &too_many).is_err());
        assert_eq!(store.snapshot().unwrap().revision, 0);
        let exhausted = LabelStore::from_snapshot(LabelSnapshot {
            identity: "test".into(),
            points: 10,
            revision: u64::MAX,
            labels: BTreeMap::new(),
        });
        assert!(exhausted.update(u64::MAX, &[update("a", vec![1])]).is_err());
        assert!(exhausted.snapshot().unwrap().labels.is_empty());
    }
}
