/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Decode generic configuration without filesystem access or dataset presets.
use super::options::IndexOptions;
use crate::{Error, Result};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

impl IndexOptions {
    /// Decode one YAML mapping using the same schema and defaults as Rust options.
    ///
    /// This is index configuration only: queries, `k`, search list size, result
    /// fields and CLI execution controls belong to separate request APIs.
    ///
    /// Missing fields retain their defaults. Paths remain relative to the caller's
    /// working directory; parsing does not open files or expand environment values.
    /// CLI adapters may override individual fields before normal validation.
    ///
    /// # Errors
    /// Returns a configuration error for malformed YAML, duplicate keys, unknown
    /// fields, multiple documents or incompatible field types. Shape-dependent and
    /// semantic validation occurs when the options are resolved by preflight or
    /// [`crate::Index::new`], after any caller overrides have been applied.
    ///
    /// ```
    /// use orion_ann::IndexOptions;
    /// let options = IndexOptions::from_yaml("threads: 8\nauto_pin: false\n")?;
    /// assert_eq!(options.threads, 8);
    /// assert!(!options.auto_pin);
    /// # Ok::<(), orion_ann::Error>(())
    /// ```
    pub fn from_yaml(input: &str) -> Result<Self> {
        // The parser's default budgets and duplicate-key rejection also apply to
        // nested object maps. Includes and environment expansion are not enabled.
        serde_saphyr::from_str::<OptionsMapping>(input)
            .map(|mapping| mapping.0)
            .map_err(|error| Error::configuration("yaml", error.to_string()).at("parse_config"))
    }
}

// Serde-derived structs can accept positional sequences and some YAML parsers
// treat an empty document as a default struct. A configuration must be a named
// field mapping: require `{}` when the caller explicitly wants all defaults.
/// Serde wrapper requiring a mapping instead of accepting scalar YAML documents.
struct OptionsMapping(IndexOptions);

impl<'de> Deserialize<'de> for OptionsMapping {
    /// Requires a YAML mapping and delegates its fields to the shared IndexOptions deserializer.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        /// Validates the outer YAML shape before delegating option fields to serde.
        struct MappingVisitor;
        impl<'de> Visitor<'de> for MappingVisitor {
            type Value = OptionsMapping;

            /// Describes the required mapping shape in serde input errors.
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an IndexOptions mapping (use {} for defaults)")
            }

            /// Deserializes the mapping through IndexOptions so field validation has one implementation.
            fn visit_map<M: MapAccess<'de>>(
                self,
                map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                IndexOptions::deserialize(de::value::MapAccessDeserializer::new(map))
                    .map(OptionsMapping)
            }
        }
        deserializer.deserialize_any(MappingVisitor)
    }
}
