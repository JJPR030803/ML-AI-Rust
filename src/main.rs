mod Modules;


use std::vec;
use Modules::dataset_struct::Dataset;
use Modules::{create_dataset::random_f64_matrix};
use rand::Rng;
use Modules::knn_ia::{self,matrix_euclidean};
use Modules::kmeans::kmeans_plus_plus;
fn main() {


    const NUM_SAMPLES:usize = 1000;
    const NUM_FEATURES:usize = 3;
    const MAX:f64 = 50.0;
    const MIN:f64 = 0.0;
    const K:usize = 3;


    //Categorias y posibles clases
    let feature_names = vec!["Height".to_string(), "Weight".to_string(), "Age".to_string()];
    let possible_classes = vec!["Class A".to_string(), "Class B".to_string(), "Class C".to_string()];


    //Generar datos aleatorios para entrenamiento
    let training_data = random_f64_matrix(NUM_FEATURES, NUM_SAMPLES, MIN, MAX);



    //Generar clases aleatorias para los datos
    let mut rng = rand::thread_rng();
    let training_labels: Vec<String> = (0..NUM_SAMPLES)
    .map(|_| possible_classes[rng.gen_range(0..possible_classes.len())].clone()
    ).collect();

    let training_dataset = Dataset::new(
        training_data.clone(),
        feature_names.clone(),
        training_labels,
    );


    //Generar datos aleatorios para testing
    let testing_data = random_f64_matrix(NUM_FEATURES, NUM_SAMPLES, MIN, MAX);
    

    //Generar clases aleatorias para testing
    let testing_labels: Vec<String> = (0..NUM_SAMPLES)
        .map(|_| possible_classes[rng.gen_range(0..possible_classes.len())].clone())
        .collect();


    //Generar dataset testing 
    let testing_dataset = Dataset::new(
        testing_data,
        feature_names.clone(),
        testing_labels
        );


        kmeans_plus_plus(&training_data, K);
        


/* 

    //Ahora si knn predicciones

    match knn_ia::evaluate_knn(&testing_dataset, &training_dataset, K) {
        Ok((accuracy,prediction)) => {
            println!("Precision del modelo KNN: {:.2}",accuracy * 100.0);
            println!("Primeras 10 predicciones: {:?}", &prediction[0..prediction.len()]);
        }

        Err(e) => println!("Erroooooor: {}",e),
        
    }
    */

}

