/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! HTTP status mapping; the response body is the unchanged public ErrorReport.
use axum::{
    extract::rejection::JsonRejection,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use orion_ann::{Error, ErrorCode};

pub(crate) struct ApiError(pub StatusCode, pub Error);
pub(crate) type ApiResult<T> = Result<T, ApiError>;

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        let status = match error.code() {
            ErrorCode::Unauthenticated => StatusCode::UNAUTHORIZED,
            ErrorCode::InvalidArgument | ErrorCode::DimensionMismatch => StatusCode::BAD_REQUEST,
            ErrorCode::InvalidData | ErrorCode::UnsupportedConfiguration => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
            ErrorCode::CacheIncompatible | ErrorCode::InvalidState => StatusCode::CONFLICT,
            ErrorCode::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self(status, error)
    }
}
impl From<JsonRejection> for ApiError {
    fn from(error: JsonRejection) -> Self {
        Self(
            error.status(),
            Error::invalid_argument(error.body_text()).at("http_decode"),
        )
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.1.report())).into_response()
    }
}
pub(crate) fn missing(kind: &str, id: &str) -> ApiError {
    ApiError(
        StatusCode::NOT_FOUND,
        Error::invalid_argument(format!("unknown {kind}"))
            .at("http_lookup")
            .with_detail("id", id),
    )
}
pub(crate) fn exhausted(message: &str) -> Error {
    Error::new(ErrorCode::ResourceExhausted, message).at("http_admission")
}

// Path decoding and router-generated rejections must also use the Orion wire
// envelope. Already structured native errors pass through byte-for-byte.
pub(crate) async fn normalize_rejection(response: Response) -> Response {
    let status = response.status();
    let json = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .is_some_and(|v| v.to_str().unwrap_or("").starts_with("application/json"));
    if (status.is_client_error() || status.is_server_error()) && !json {
        ApiError(
            status,
            Error::invalid_argument(status.canonical_reason().unwrap_or("HTTP request rejected"))
                .at("http_decode"),
        )
        .into_response()
    } else {
        response
    }
}
