/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use clap::Parser;
use orion_ann::Error;
use orion_ann_server::{router, ServerConfig, TokenAuthenticator};
use std::{net::SocketAddr, path::PathBuf};

#[derive(Debug, Parser)]
#[command(about = "Local Orion native HTTP service")]
struct Args {
    /// JSON array of subject and token_sha256 records; never pass keys in argv.
    #[arg(long)]
    auth_file: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    #[arg(long, default_value = "orion-service-data")]
    data_root: PathBuf,
    #[arg(long, default_value_t = 1)]
    build_workers: usize,
    #[arg(long, default_value_t = 8)]
    max_indexes: usize,
    #[arg(long, default_value_t = 8)]
    max_searches: usize,
    #[arg(long, default_value_t = 64)]
    max_threads: usize,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run(Args::parse()).await {
        eprintln!("{}", serde_json::to_string(&error.report()).unwrap());
        std::process::exit(1);
    }
}
async fn run(args: Args) -> orion_ann::Result<()> {
    if !args.listen.ip().is_loopback() {
        return Err(Error::invalid_argument(
            "prototype listener must bind to loopback",
        ));
    }
    let app = router(ServerConfig {
        data_root: args.data_root,
        authenticator: Some(std::sync::Arc::new(TokenAuthenticator::from_file(
            args.auth_file,
        )?)),
        build_workers: args.build_workers,
        max_indexes: args.max_indexes,
        max_searches: args.max_searches,
        max_threads: args.max_threads,
        ..Default::default()
    })?;
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .map_err(Error::from)?;
    println!("Orion HTTP listening on http://{}", listener.local_addr()?);
    // Blocking native tasks finish during shutdown. This is not hard cancellation.
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(Error::from)
}
