/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Credential verification only. Resource ownership is enforced separately.
use crate::error::ApiError;
use axum::{
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use orion_ann::{Error, ErrorCode, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fmt, future::Future, path::Path, pin::Pin, sync::Arc};
use subtle::ConstantTimeEq;

/// Stable, server-verified resource owner; never taken from a request body/header.
/// Changing a subject changes its storage namespace. Reuse it across key rotation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Principal {
    subject: String,
}
impl Principal {
    /// Subject names are safe single directory components (ASCII, 1..=64 bytes).
    pub fn new(subject: impl Into<String>) -> Result<Self> {
        let subject = subject.into();
        if subject.is_empty()
            || subject.len() > 64
            || !subject
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        {
            return Err(Error::invalid_argument("invalid authentication subject"));
        }
        Ok(Self { subject })
    }
    pub fn subject(&self) -> &str {
        &self.subject
    }
}

/// Async boundary for configuration, database or OIDC implementations. `None`
/// means rejected credentials; `Err` means verifier failure. Never include the
/// credential in an error or Debug output. Implementations must verify credentials
/// before issuing a Principal, and must not block the async executor on I/O.
pub trait Authenticator: fmt::Debug + Send + Sync + 'static {
    fn authenticate<'a>(
        &'a self,
        bearer: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Principal>>> + Send + 'a>>;
}

/// Operator-managed bearer keys. Stores SHA-256 digests, not plaintext keys.
/// Keys must be random high-entropy secrets, not human passwords. SHA-256 is not
/// a password KDF. Reload/restart the service to change the configured keys.
#[derive(Clone, Default)]
pub struct TokenAuthenticator {
    entries: Vec<(Principal, [u8; 32])>,
}
impl fmt::Debug for TokenAuthenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenAuthenticator")
            .field("credentials", &self.entries.len())
            .finish_non_exhaustive()
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Credential {
    subject: String,
    token_sha256: String,
}
impl TokenAuthenticator {
    /// Parse an array of `{subject, token_sha256}` records. Duplicate keys and
    /// malformed digests are rejected; multiple keys may share a subject.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::io("http_credentials", path, e))?;
        let records: Vec<Credential> = serde_json::from_slice(&bytes)
            .map_err(|_| Error::invalid_argument("invalid authentication configuration"))?;
        let mut result = Self::default();
        for record in records {
            let principal = Principal::new(record.subject)?;
            let mut digest = [0; 32];
            if record.token_sha256.len() != 64 {
                return Err(Error::invalid_argument("invalid credential digest"));
            }
            for (dest, hex) in digest
                .iter_mut()
                .zip(record.token_sha256.as_bytes().chunks_exact(2))
            {
                let text = std::str::from_utf8(hex)
                    .map_err(|_| Error::invalid_argument("invalid credential digest"))?;
                *dest = u8::from_str_radix(text, 16)
                    .map_err(|_| Error::invalid_argument("invalid credential digest"))?;
            }
            if result.entries.iter().any(|(_, d)| d == &digest) {
                return Err(Error::invalid_argument("duplicate credential digest"));
            }
            result.entries.push((principal, digest));
        }
        if result.entries.is_empty() {
            return Err(Error::invalid_argument(
                "authentication configuration is empty",
            ));
        }
        Ok(result)
    }
}
impl Authenticator for TokenAuthenticator {
    fn authenticate<'a>(
        &'a self,
        bearer: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Principal>>> + Send + 'a>> {
        Box::pin(async move {
            let digest: [u8; 32] = Sha256::digest(bearer.as_bytes()).into();
            let mut found = None;
            // Always compare every digest. No early-exit prefix comparison of keys.
            for (principal, expected) in &self.entries {
                if bool::from(digest.ct_eq(expected)) {
                    found = Some(principal.clone());
                }
            }
            Ok(found)
        })
    }
}

pub(crate) async fn authenticate(
    State(auth): State<Arc<dyn Authenticator>>,
    mut request: Request,
    next: Next,
) -> Response {
    if request.method() == axum::http::Method::GET && request.uri().path() == "/health" {
        return next.run(request).await;
    }
    let mut headers = request
        .headers()
        .get_all(axum::http::header::AUTHORIZATION)
        .iter();
    let bearer = headers.next().and_then(|h| h.to_str().ok()).and_then(|h| {
        let (scheme, token) = h.split_once(' ')?;
        (scheme.eq_ignore_ascii_case("Bearer")
            && !token.is_empty()
            && token.len() <= 4096
            && token.bytes().all(|b| b.is_ascii_graphic()))
        .then_some(token)
    });
    let verified = match (bearer, headers.next()) {
        (Some(token), None) => {
            match tokio::time::timeout(std::time::Duration::from_secs(5), auth.authenticate(token))
                .await
            {
                Ok(Ok(principal)) => principal,
                // The provider's error may contain connection/credential details.
                // Keep those out of public responses and logs.
                _ => {
                    return ApiError(
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        Error::internal("authentication service unavailable")
                            .at("http_authentication"),
                    )
                    .into_response()
                }
            }
        }
        _ => None,
    };
    match verified {
        Some(principal) => {
            request.extensions_mut().insert(principal.clone());
            let mut response = next.run(request).await;
            response.extensions_mut().insert(principal);
            response
        }
        None => {
            let mut response = ApiError(
                axum::http::StatusCode::UNAUTHORIZED,
                Error::new(
                    ErrorCode::Unauthenticated,
                    "valid bearer credentials required",
                )
                .at("http_authentication"),
            )
            .into_response();
            response.headers_mut().insert(
                axum::http::header::WWW_AUTHENTICATE,
                axum::http::HeaderValue::from_static("Bearer realm=\"orion\""),
            );
            response
        }
    }
}
