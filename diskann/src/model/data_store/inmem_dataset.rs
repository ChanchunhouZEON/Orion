/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use rayon::prelude::*;
use std::mem;
use vector::{FullPrecisionDistance, Metric};

use crate::common::{ANNError, ANNResult, AlignedBoxWithSlice};
use crate::model::Vertex;
use crate::utils::copy_aligned_data_from_file;

#[derive(Debug)]
pub struct InmemDataset<T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    pub data: AlignedBoxWithSlice<T>,
    pub num_points: usize,
    pub num_active_pts: usize,
    pub capacity: usize,
}

impl<'a, T, const N: usize> InmemDataset<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    pub fn new(num_points: usize, index_growth_factor: f32) -> ANNResult<Self> {
        let capacity = (((num_points * N) as f32) * index_growth_factor) as usize;

        Ok(Self {
            data: AlignedBoxWithSlice::new(capacity, mem::size_of::<T>() * 16)?,
            num_points,
            num_active_pts: num_points,
            capacity,
        })
    }

    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    pub fn build_from_file(&mut self, filename: &str, num_points_to_load: usize) -> ANNResult<()> {
        println!(
            "Loading {} vectors from file {} into dataset...",
            num_points_to_load, filename
        );
        self.num_active_pts = num_points_to_load;

        copy_aligned_data_from_file(filename, self.into_dto(), 0)?;

        println!("Dataset loaded.");
        Ok(())
    }

    pub fn append_from_file(
        &mut self,
        filename: &str,
        num_points_to_append: usize,
    ) -> ANNResult<()> {
        println!(
            "Appending {} vectors from file {} into dataset...",
            num_points_to_append, filename
        );
        if self.num_points + num_points_to_append > self.capacity {
            return Err(ANNError::log_index_error(format!(
                "Cannot append {} points to dataset of capacity {}",
                num_points_to_append, self.capacity
            )));
        }

        let pts_offset = self.num_active_pts;
        copy_aligned_data_from_file(filename, self.into_dto(), pts_offset)?;

        self.num_active_pts += num_points_to_append;
        self.num_points += num_points_to_append;

        println!("Dataset appended.");
        Ok(())
    }

    pub fn get_vertex(&'a self, id: u32) -> ANNResult<Vertex<'a, T, N>> {
        let start = id as usize * N;
        let end = start + N;

        if end <= self.data.len() {
            let val = <&[T; N]>::try_from(&self.data[start..end]).map_err(|err| {
                ANNError::log_index_error(format!("Failed to get vertex {}, err={}", id, err))
            })?;
            Ok(Vertex::new(val, id))
        } else {
            Err(ANNError::log_index_error(format!(
                "Invalid vertex id {}.",
                id
            )))
        }
    }

    pub fn get_distance(&self, id1: u32, id2: u32, metric: Metric) -> ANNResult<f32> {
        let vertex1 = self.get_vertex(id1)?;
        let vertex2 = self.get_vertex(id2)?;

        Ok(vertex1.compare(&vertex2, metric))
    }

    pub fn calculate_medoid_point_id(&self) -> ANNResult<u32> {
        Ok(self.find_nearest_point_id(self.calculate_centroid_point()?))
    }

    fn calculate_centroid_point(&self) -> ANNResult<[f32; N]> {
        let mut center: [f32; N] = [0.0; N];

        for i in 0..self.num_active_pts {
            let vertex = self.get_vertex(i as u32)?;
            let vertex_slice = vertex.vector();
            for j in 0..N {
                center[j] += vertex_slice[j].into();
            }
        }

        let capacity = self.num_active_pts as f32;
        for item in center.iter_mut().take(N) {
            *item /= capacity;
        }

        Ok(center)
    }

    fn find_nearest_point_id(&self, point: [f32; N]) -> u32 {
        let mut distances = vec![0f32; self.num_active_pts];
        let slice = &self.data[..];
        distances.par_iter_mut().enumerate().for_each(|(i, dist)| {
            let start = i * N;
            for j in 0..N {
                let diff: f32 = (point.as_slice()[j] - slice[start + j].into())
                    * (point.as_slice()[j] - slice[start + j].into());
                *dist += diff;
            }
        });

        let mut min_idx = 0;
        let mut min_dist = f32::MAX;
        for (i, distance) in distances.iter().enumerate().take(self.num_active_pts) {
            if *distance < min_dist {
                min_idx = i;
                min_dist = *distance;
            }
        }
        min_idx as u32
    }

    #[inline]
    pub fn prefetch_vector(&self, id: u32) {
        let start = id as usize * N;
        let end = start + N;

        if end <= self.data.len() {
            let vec = &self.data[start..end];
            vector::prefetch_vector(vec);
        }
    }

    pub fn into_dto(&mut self) -> DatasetDto<T> {
        DatasetDto {
            data: &mut self.data,
            rounded_dim: N,
        }
    }
}

#[derive(Debug)]
pub struct DatasetDto<'a, T> {
    pub data: &'a mut [T],
    pub rounded_dim: usize,
}
