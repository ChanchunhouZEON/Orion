/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

mod data;
mod load_log;
mod run;
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{fs::File, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Estimate {
        #[arg(long)]
        dataset: PathBuf,
    },
    Calibrate {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        work: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Run(run::Args),
    Fixture {
        #[arg(long)]
        directory: PathBuf,
    },
}
#[derive(Clone, Serialize, Deserialize)]
struct Calibration {
    k: usize,
    window: usize,
    epsilon: f32,
    early_exit_limit: usize,
    beam: usize,
    samples: usize,
}
fn main() -> Result<()> {
    let args = Args::parse();
    // Calibration uses Rayon; timed single requests use the caller directly.
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()?;
    match args.command {
        Command::Estimate { dataset } => println!("{}", data::Dataset::load(&dataset)?.estimate()?),
        Command::Calibrate {
            dataset,
            work,
            output,
        } => {
            std::fs::create_dir_all(&work)?;
            let d = data::Dataset::load(&dataset)?;
            let o = data::mapped_options(&work);
            let base = data::base(&d.base, &o)?;
            let index = data::index(&d, base, &o)?;
            let (dim, q) = data::vectors::<f32>(&d.queries)?;
            let samples = (q.len() / dim).min(100);
            let beam = *d.beams.iter().max().unwrap();
            let p = index.calibrate(
                &q[..samples * dim],
                beam,
                3,
                orion_ann_core::CalibrationConfig::default(),
            )?;
            let c = Calibration {
                k: 10,
                window: 3,
                epsilon: p.threshold,
                early_exit_limit: p.early_exit_limit,
                beam,
                samples,
            };
            serde_json::to_writer_pretty(File::create(output)?, &c)?;
        }
        Command::Run(args) => run::execute(args)?,
        Command::Fixture { directory } => fixture(directory)?,
    }
    Ok(())
}
fn fixture(dir: PathBuf) -> Result<()> {
    use orion_ann_core::model::dataset::{
        JLSparseDataset, L2KTDataset, ScalarQuantizedDataset, L2U8,
    };
    use orion_ann_core::{Orion, QuantizedDataset};
    use orion_ann_diskann::model::PaddedDataset;
    use orion_ann_vector::{L2F32Distance, PaddedLayout};
    use std::io::{BufWriter, Write};
    std::fs::create_dir_all(&dir)?;
    let n = 256;
    let dim = 128;
    let values: Vec<f32> = (0..n * dim)
        .map(|i| ((i * 31 + i / dim * 7) % 251) as f32)
        .collect();
    let base = PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<L2F32Distance>(dim).map_err(anyhow::Error::msg)?,
    )
    .map_err(anyhow::Error::msg)?;
    fn vecs(path: PathBuf, rows: &[f32], d: usize) -> Result<()> {
        let mut f = File::create(path)?;
        for row in rows.chunks(d) {
            f.write_all(&(d as u32).to_le_bytes())?;
            f.write_all(bytemuck::cast_slice(row))?;
        }
        Ok(())
    }
    vecs(dir.join("base.fvecs"), &values, dim)?;
    vecs(dir.join("query.fvecs"), &values[..32 * dim], dim)?;
    let mut gt = File::create(dir.join("truth.ivecs"))?;
    for query in values[..32 * dim].chunks(dim) {
        let mut sorted: Vec<_> = values
            .chunks(dim)
            .enumerate()
            .map(|(i, r)| {
                (
                    r.iter()
                        .zip(query)
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f32>(),
                    i as u32,
                )
            })
            .collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        gt.write_all(&10u32.to_le_bytes())?;
        for (_, id) in sorted.into_iter().take(10) {
            gt.write_all(&id.to_le_bytes())?;
        }
    }
    let scalar = ScalarQuantizedDataset::<L2U8>::build(&base, &())?;
    scalar.encode(&mut File::create(dir.join("scalar.qds"))?)?;
    let kt = L2KTDataset::build(&base, &())?;
    kt.encode(&mut File::create(dir.join("kt.qds"))?)?;
    let jl =
        JLSparseDataset::<1024>::build(&base, &JLSparseDataset::<1024>::default_build_options())?;
    jl.encode(&mut File::create(dir.join("jl.qds"))?)?;
    let parts: Vec<_> = (0..n as u32)
        .map(|i| {
            (
                (1..9).map(|j| (i + j * 7) % n as u32).collect(),
                (9..16).map(|j| (i + j * 7) % n as u32).collect(),
                vec![(i + 1) % n as u32],
            )
        })
        .collect();
    let index = Orion::from_graph_partitions(base, &parts, 0, 16, 1)?;
    index.encode_native_graph(
        File::create(dir.join("metadata.bin"))?,
        &mut BufWriter::new(File::create(dir.join("graph.bin"))?),
    )?;
    for (name, scalar, kt, jl) in [
        ("sift", Some("scalar.qds"), None, None),
        ("gist", None, Some("kt.qds"), Some("jl.qds")),
    ] {
        let d = data::Dataset {
            name: name.into(),
            base: "base.fvecs".into(),
            queries: "query.fvecs".into(),
            ground_truth: "truth.ivecs".into(),
            graph: "graph.bin".into(),
            metadata: "metadata.bin".into(),
            scalar: scalar.map(Into::into),
            kt: kt.map(Into::into),
            jl: jl.map(Into::into),
            beams: vec![32, 48],
        };
        serde_json::to_writer_pretty(File::create(dir.join(format!("{name}.json")))?, &d)?;
    }
    let c = Calibration {
        k: 10,
        window: 3,
        epsilon: 0.5,
        early_exit_limit: 4,
        beam: 48,
        samples: 32,
    };
    ensure!(!index.dataset().as_slice().is_empty(), "empty fixture");
    serde_json::to_writer_pretty(File::create(dir.join("calibration.json"))?, &c)?;
    Ok(())
}
