use std::collections::HashMap;

use super::{dataset_struct::Dataset, distancias::euclidean_distance};
use rand::Rng;

/// Initializes centroids using the KMeans++ algorithm
///
/// # Algorithm Overview:
/// KMeans++ improves on the standard KMeans initialization by selecting centroids
/// that are far away from each other, which often leads to better clustering results.
/// The algorithm:
/// 1. Randomly selects the first centroid from the dataset
/// 2. For each subsequent centroid:
///    a. Calculates the minimum distance from each data point to any existing centroid
///    b. Selects the next centroid with probability proportional to these distances
///       (points farther from existing centroids have higher probability of selection)
///
/// # Parameters:
/// - `data`: &Vec<Vec<f64>> - The dataset to calculate centroids from, where each inner vector
///   represents a data point with multiple dimensions/features
/// - `k`: usize - The number of clusters/centroids to generate
///
/// # Returns:
/// - Vec<Vec<f64>> - A vector containing k centroids, where each centroid is a vector of f64 values
///   representing a point in the same dimensional space as the input data
///
/// # Notes:
/// - The function handles edge cases (empty data, invalid k values)
/// - Includes safeguards against numerical issues like division by zero
pub fn kmeans_plus_plus(data: &Vec<Vec<f64>>, k: usize) -> Vec<Vec<f64>> {
    if data.is_empty() || k == 0 || k > data.len() {
        return Vec::new();
    }

    let mut rng = rand::thread_rng();
    let mut centroids = Vec::with_capacity(k);

    // Randomly select the first centroid from the dataset
    centroids.push(data[rng.gen_range(0..data.len())].clone());

    // Continue selecting centroids until we have k centroids
    while centroids.len() < k {
        // For each data point, calculate the minimum distance to any existing centroid
        let mut distances: Vec<f64> = data
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|c| euclidean_distance(point, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        // Prevent division by zero by checking total distance
        let total_distance: f64 = distances.iter().sum();
        if total_distance <= 0.0 {
            // If all distances are zero, select a random point as a centroid
            centroids.push(data[rng.gen_range(0..data.len())].clone());
            continue;
        }

        // Select next centroid with probability proportional to squared distance
        // to the closest existing centroid
        let mut cumulative_prob = 0.0;
        let target = rng.gen::<f64>() * total_distance;
        let mut selected = false;

        for (i, &d) in distances.iter().enumerate() {
            cumulative_prob += d;
            if cumulative_prob >= target {
                centroids.push(data[i].clone());
                selected = true;
                break;
            }
        }

        // Safeguard against floating-point issues that might prevent selection
        if !selected {
            centroids.push(data[rng.gen_range(0..data.len())].clone());
        }
    }

    println!("Initial centroids selected: {} centroids", centroids.len());

    centroids
}

/// Assigns each data point to the nearest cluster using Euclidean distance
///
/// # Algorithm Overview:
/// For each data point, this function calculates the Euclidean distance to each centroid,
/// and assigns the point to the cluster with the nearest centroid.
///
/// # Parameters:
/// - `data`: Vec<Vec<f64>> - The dataset to assign to clusters
/// - `centroides`: Vec<Vec<f64>> - The current centroids of each cluster
///
/// # Returns:
/// - Vec<usize> - A vector of cluster indices (0 to k-1) for each data point in the same order
///   as the input data. Each value indicates which cluster the corresponding data point belongs to.
///
/// # Notes:
/// - Returns an empty vector if either input is empty
/// - Uses min_by with partial_cmp to handle potential floating-point comparison issues
pub fn asignar_a_clusters(data: Vec<Vec<f64>>, centroides: &Vec<Vec<f64>>) -> Vec<usize> {
    if data.is_empty() || centroides.is_empty() {
        return Vec::new();
    }

    data.iter()
        .map(|point| {
            centroides
                .iter()
                .enumerate()
                .map(|(i, centroid)| (i, euclidean_distance(point, centroid)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0) // Failsafe if min_by returns None
        })
        .collect()
}

/// Updates centroids by calculating the mean of all points assigned to each cluster
///
/// # Algorithm Overview:
/// This function:
/// 1. Initializes new centroids as zeros
/// 2. For each data point, adds its values to the corresponding cluster's centroid
/// 3. Counts how many points are assigned to each cluster
/// 4. Divides each centroid by its count to get the mean position
///
/// # Parameters:
/// - `data`: &Vec<Vec<f64>> - The dataset
/// - `clusters`: &Vec<usize> - The cluster assignments for each data point
/// - `k`: usize - The number of clusters
///
/// # Returns:
/// - Vec<Vec<f64>> - Updated centroids based on the mean of points in each cluster
///
/// # Notes:
/// - Handles edge cases like empty data, invalid cluster assignments
/// - Warns if a centroid has no points assigned to it
/// - Maintains centroids with zero points assigned to avoid numerical issues
pub fn actualizar_centroides(
    data: &Vec<Vec<f64>>,
    clusters: &Vec<usize>,
    k: usize,
) -> Vec<Vec<f64>> {
    if data.is_empty() || clusters.is_empty() || data.len() != clusters.len() {
        return vec![vec![0.0; data.get(0).map_or(0, |v| v.len())]; k];
    }

    let dim = data[0].len();
    let mut nuevos_centroides = vec![vec![0.0; dim]; k];
    let mut counts = vec![0; k];

    // Sum up all points assigned to each cluster
    for (point, &cluster) in data.iter().zip(clusters.iter()) {
        if cluster < k {
            // Validate cluster index is in bounds
            for (i, &value) in point.iter().enumerate() {
                if i < dim {
                    // Validate dimension index is in bounds
                    nuevos_centroides[cluster][i] += value;
                }
            }
            counts[cluster] += 1;
        }
    }

    // Calculate the mean position for each cluster
    for (i, centroid) in nuevos_centroides.iter_mut().enumerate() {
        if counts[i] > 0 {
            for value in centroid.iter_mut() {
                *value /= counts[i] as f64;
            }
        } else {
            // If a centroid has no points assigned, keep it unchanged
            // (This can happen and may indicate a suboptimal k value)
            println!("Warning: Centroid {} has no assigned points", i);
        }
    }

    nuevos_centroides
}

/// Iterates the KMeans algorithm until convergence or maximum iterations are reached
///
/// # Algorithm Overview:
/// The KMeans algorithm works by:
/// 1. Initializing centroids using KMeans++
/// 2. Repeating until convergence or max iterations:
///    a. Assign each data point to the nearest centroid
///    b. Update centroids to be the mean of all points in the cluster
///    c. Check if centroids have changed significantly
///
/// # Parameters:
/// - `max_iters`: usize - Maximum number of iterations
/// - `data`: Vec<Vec<f64>> - The dataset to cluster
/// - `k`: usize - The number of clusters
///
/// # Returns:
/// - Vec<Vec<f64>> - Final centroids after convergence or max iterations
/// Explicacion por que se me olvida:
/// Regresa algo similar a coordenadas (x,y) si son 2 clases o columnas
/// (X,Y,Z) si son 3 columnas y asi consecutivamente
///
/// # Notes:
/// - Convergence is checked by comparing old and new centroids
/// - A small threshold (1e-6) is used to account for floating-point precision
/// - The algorithm can terminate early if centroids stop changing
pub fn iterar_hasta_optimo(max_iters: usize, data: Vec<Vec<f64>>, k: usize) -> Vec<Vec<f64>> {
    if data.is_empty() || k == 0 || k > data.len() {
        return Vec::new();
    }

    // Initialize centroids using KMeans++
    let mut centroids = kmeans_plus_plus(&data, k);
    let mut iteration = 0;

    // Main KMeans iteration loop
    while iteration < max_iters {
        // Assign data points to nearest centroid
        let clusters = asignar_a_clusters(data.clone(), &centroids);

        // Update centroids based on cluster assignments
        let new_centroids = actualizar_centroides(&data, &clusters, k);

        // Check for convergence (centroids no longer moving significantly)
        let mut converged = true;
        for (i, centroid) in centroids.iter().enumerate() {
            for (j, &value) in centroid.iter().enumerate() {
                if (value - new_centroids[i][j]).abs() > 1e-6 {
                    converged = false;
                    break;
                }
            }
            if !converged {
                break;
            }
        }

        // If converged, exit early
        if converged {
            println!("KMeans converged after {} iterations", iteration + 1);
            break;
        }

        // Otherwise, update centroids and continue
        centroids = new_centroids;
        iteration += 1;
    }

    // Notify if reached maximum iterations without convergence
    if iteration == max_iters {
        println!(
            "KMeans reached maximum iterations ({}) without converging",
            max_iters
        );
    }

    centroids
}

///Asigna centroides a labels o categorias
/// #Parametros:
/// -centroidss(x,y) si son 2 clases
/// -labels: etiquetas para categorizar
///
/// #Return
/// Hashmap<usize,String>
///

// Function to map centroids to custom labels
pub fn map_centroids_to_labels(
    centroids: &Vec<Vec<f64>>,
    labels: &Vec<String>,
) -> HashMap<usize, String> {
    let mut centroid_labels = HashMap::new();

    // Assign each centroid an index and corresponding label
    for (i, _) in centroids.iter().enumerate() {
        let label = if i < labels.len() {
            labels[i].clone()
        } else {
            // Handle case where there are more centroids than labels
            format!("Cluster_{}", i)
        };

        centroid_labels.insert(i, label);
    }

    centroid_labels
}

///Funcion para obtener la categoria donde esta el punto dado
///
// Function to get the label for a data point based on its assigned cluster
pub fn get_point_label(
    point_idx: usize,
    clusters: &Vec<usize>,
    centroid_labels: &HashMap<usize, String>,
) -> String {
    let cluster_idx = clusters[point_idx];
    centroid_labels
        .get(&cluster_idx)
        .unwrap_or(&format!("Unknown_{}", cluster_idx))
        .clone()
}

pub fn show_cluster_info(centroides: &Vec<Vec<f64>>, centroid_labels: &HashMap<usize, String>) {
    for (i, centroid) in centroides.iter().enumerate() {
        let label = centroid_labels.get(&i).unwrap();
        println!(
            "Cluster {}: Label '{}' - Centroid: {:?}",
            i, label, centroid
        );
    }
}

pub fn show_sample_cluster(
    training_data: &Vec<Vec<f64>>,
    centroid_labels: &HashMap<usize, String>,
    point_clusters: &Vec<usize>,
    n_samples: usize
) {
    println!("\nSample of data points with assigned cluster labels:");
    for i in 0..n_samples.min(training_data.len()) {
        let point = &training_data[i];
        let cluster_idx = point_clusters[i];
        let label = centroid_labels.get(&cluster_idx).unwrap();
        println!(
            "Point {:?} - Assigned to cluster {} (Label '{}')",
            point, cluster_idx, label
        );
    }
}

pub fn show_cluster_stats(
    cluster_counts: &HashMap<usize, i32>,
    centroid_labels: &HashMap<usize, String>,
    point_clusters: &Vec<usize>,
) {
    // Print cluster statistics
    println!("\nCluster Statistics:");
    for (cluster_idx, count) in cluster_counts.iter() {
        let label = centroid_labels.get(cluster_idx).unwrap();
        let percentage = (*count as f64 / point_clusters.len() as f64) * 100.0;
        println!(
            "Cluster {} (Label '{}'): {} points ({:.2}%)",
            cluster_idx, label, count, percentage
        );
    }
}
