mod modules;
use modules::create_dataset::random_f64_matrix;
use modules::dataset_struct::Dataset;
use modules::kmeans::{
    actualizar_centroides, asignar_a_clusters, iterar_hasta_optimo, kmeans_plus_plus,
    map_centroids_to_labels, show_cluster_info, show_cluster_stats, show_sample_cluster,
};
use modules::*;
use rand::Rng;
use std::collections::HashMap;
use std::vec;

fn main() {
    const NUM_SAMPLES: usize = 1000;
    const NUM_FEATURES: usize = 2;
    const MAX: f64 = 100.0;
    const MIN: f64 = 0.0;
    const K: usize = 3; // Set to match the actual number of clusters used

    // Feature names
    let feature_names = vec!["X".to_string(), "Y".to_string()];

    // Custom cluster labels
    let cluster_labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];

    // Generate random dataset
    let (testing_dataset, training_dataset, _) = Dataset::initialize_datasets(
        NUM_FEATURES,
        NUM_SAMPLES,
        MIN,
        MAX,
        feature_names,
        cluster_labels.clone(),
    );

    let training_data = training_dataset.rows.clone();

    // Run KMeans clustering
    let final_centroids = iterar_hasta_optimo(100, training_data.clone(), K);

    // Create a mapping from centroid index to custom label
    let centroid_labels = map_centroids_to_labels(&final_centroids, &cluster_labels);

    // Assign each data point to a cluster
    let point_clusters = asignar_a_clusters(training_data.clone(), &final_centroids);

    show_cluster_info(&final_centroids, &centroid_labels);

    // Count points in each cluster
    let mut cluster_counts: HashMap<usize, i32> = HashMap::new();
    for &cluster in &point_clusters {
        *cluster_counts.entry(cluster).or_insert(0) += 1;
    }

    show_cluster_stats(&cluster_counts, &centroid_labels, &point_clusters);

    show_sample_cluster(&training_data, &centroid_labels, &point_clusters,20);
}
