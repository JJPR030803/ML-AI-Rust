mod modules;

use std::vec;
use modules::dataset_struct::Dataset;
use modules::create_dataset::random_f64_matrix;
use rand::Rng;
use modules::kmeans::{kmeans_plus_plus, asignar_a_clusters, actualizar_centroides, iterar_hasta_optimo};
use modules::graphs::{plot_centroides, plot_3d}; 

fn main() {
    const NUM_SAMPLES: usize = 1000;
    const NUM_FEATURES: usize = 3;
    const MAX: f64 = 100.0;
    const MIN: f64 = 0.0;
    const K: usize = 3;
    // New parameter to control whether to show class info
    const SHOW_CLASSES: bool = false;

    let (testing_dataset, training_dataset, feature_names) = inicializar_datasets(NUM_FEATURES, NUM_SAMPLES, MIN, MAX);
    let training_data = training_dataset.rows.clone();
    let testing_data = testing_dataset.rows.clone();

    // K-means clustering initialization
    println!("Initializing K-means++ with K={}", K);
    let centroides = kmeans_plus_plus(&training_data, K);
    let cluster_assignments = asignar_a_clusters(training_data.clone(), centroides.clone());

    println!("Number of cluster assignments: {:?}", cluster_assignments.len());

    // Plot initial centroids and cluster assignments (2D)
    if let Err(e) = plot_centroides(
        training_dataset.rows.clone(), 
        &centroides, 
        MIN, 
        MAX, 
        "centroide_original.png",
        Some(&cluster_assignments),
        SHOW_CLASSES,
        Some("Initial K-means++ Cluster Assignments"),
    ) {
        eprintln!("Error plotting initial centroids: {}", e);
    }
    
    // Plot initial centroids and cluster assignments (3D)
    if let Err(e) = plot_3d(
        &training_dataset.rows, 
        &centroides, 
        MIN, 
        MAX, 
        "centroide_original_3d.png", 
        &feature_names,
        Some(&cluster_assignments),
        SHOW_CLASSES,
        Some("Initial K-means++ Cluster Assignments"),
    ) {
        eprintln!("Error plotting 3D initial centroids: {}", e);
    }

    // Update centroids once
    let centroides_actualizados = actualizar_centroides(&training_data, &cluster_assignments, K);
    // Get new assignments after updating centroids
    let cluster_assignments_updated = asignar_a_clusters(training_data.clone(), centroides_actualizados.clone());

    // Plot updated centroids and cluster assignments (2D)
    if let Err(e) = plot_centroides(
        training_dataset.rows.clone(), 
        &centroides_actualizados, 
        MIN, 
        MAX, 
        "centroide_actualizado.png",
        Some(&cluster_assignments_updated),
        SHOW_CLASSES,
        Some("Updated K-means++ Cluster Assignments (1 iteration)"),
    ) {
        eprintln!("Error plotting updated centroids: {}", e);
    }
    
    // Plot updated centroids and cluster assignments (3D)
    if let Err(e) = plot_3d(
        &training_dataset.rows, 
        &centroides_actualizados, 
        MIN, 
        MAX, 
        "centroide_actualizado_3d.png", 
        &feature_names,
        Some(&cluster_assignments_updated),
        SHOW_CLASSES,
        Some("Updated K-means++ Cluster Assignments (1 iteration)"),
    ) {
        eprintln!("Error plotting 3D updated centroids: {}", e);
    }

    // Iterate until convergence
    println!("Starting K-means iteration until convergence...");
    let centroides_finales = iterar_hasta_optimo(100, training_dataset.rows.clone(), K);
    // Get final assignments
    let cluster_assignments_final = asignar_a_clusters(training_data.clone(), centroides_finales.clone());
    
    // Plot final centroids and cluster assignments (2D)
    if let Err(e) = plot_centroides(
        training_dataset.rows.clone(), 
        &centroides_finales, 
        MIN, 
        MAX, 
        "kmeans_final.png",
        Some(&cluster_assignments_final),
        SHOW_CLASSES,
        Some("Final K-means++ Cluster Assignments"),
    ) {
        eprintln!("Error plotting final centroids: {}", e);
    }
    
    // Plot final centroids and cluster assignments (3D)
    if let Err(e) = plot_3d(
        &training_dataset.rows, 
        &centroides_finales, 
        MIN, 
        MAX, 
        "kmeans_final_3d.png", 
        &feature_names,
        Some(&cluster_assignments_final),
        SHOW_CLASSES,
        Some("Final K-means++ Cluster Assignments"),
    ) {
        eprintln!("Error plotting 3D final centroids: {}", e);
    }
    
    // Print final centroids and some statistics for analysis
    println!("Final centroids:");
    for (i, centroid) in centroides_finales.iter().enumerate() {
        println!("Centroid {}: {:?}", i, centroid);
        
        // Count how many points are assigned to this centroid
        let count = cluster_assignments_final.iter()
            .filter(|&&assignment| assignment == i)
            .count();
        
        println!("  Points assigned to cluster {}: {} ({}%)", 
            i, 
            count, 
            (count as f64 / cluster_assignments_final.len() as f64 * 100.0).round()
        );
    }

    // Optionally plot the testing dataset with the same centroids
    if let Err(e) = plot_centroides(
        testing_dataset.rows.clone(), 
        &centroides_finales, 
        MIN, 
        MAX, 
        "testing_with_final_centroids.png",
        None, // Just show testing data points without cluster assignments
        SHOW_CLASSES,
        Some("Testing Data with Final Centroids"),
    ) {
        eprintln!("Error plotting testing data: {}", e);
    }
}


//Para limpiar main
fn inicializar_datasets(NUM_FEATURES:usize,NUM_SAMPLES:usize,MIN:f64,MAX:f64)-> (Dataset,Dataset,Vec<String>){

    // Generar datos aleatorios para entrenamiento
    let training_data = random_f64_matrix(NUM_FEATURES, NUM_SAMPLES, MIN, MAX);

     // Categorias y posibles clases
     let feature_names = vec!["Height".to_string(), "Weight".to_string(), "Age".to_string()];
     let possible_classes = vec!["Class A".to_string(), "Class B".to_string(), "Class C".to_string()];

      // Generar clases aleatorias para los datos
    let mut rng = rand::thread_rng();
    let training_labels: Vec<String> = (0..NUM_SAMPLES)
        .map(|_| possible_classes[rng.gen_range(0..possible_classes.len())].clone())
        .collect();

    let training_dataset = Dataset::new(
        training_data.clone(),
        feature_names.clone(),
        training_labels,
    );

    // Generar datos aleatorios para testing
    let testing_data = random_f64_matrix(NUM_FEATURES, NUM_SAMPLES, MIN, MAX);
    
    // Generar clases aleatorias para testing
    let testing_labels: Vec<String> = (0..NUM_SAMPLES)
        .map(|_| possible_classes[rng.gen_range(0..possible_classes.len())].clone())
        .collect();

    // Generar dataset testing 
    let testing_dataset = Dataset::new(
        testing_data,
        feature_names.clone(),
        testing_labels
    );

    (testing_dataset,training_dataset,feature_names)
 
}