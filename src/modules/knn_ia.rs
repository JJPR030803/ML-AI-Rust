use std::{fmt::format, vec};
use super::dataset_struct::Dataset;
use super::distancias::euclidean_distance;


pub fn matrix_euclidean(
    target_dataset: &Dataset,
    training_dataset: &Dataset,
    k:usize
)->
Result<Vec<Vec<(usize,f64,String)>>,String>
 {

    //Verificar el numero de features de cada dataset(training,testing)
    if !target_dataset.rows.is_empty() && !training_dataset.rows.is_empty(){
        let target_features = target_dataset.rows[0].len();
        let training_features = training_dataset.rows[0].len();

        if target_features != training_features{
            return Err(format!(
                "Conteo de features no coinciden: dataset de testing tiene {} features, dataset de entrenamiento tiene {}features",
                target_features,training_features
            ));
        }
    }

    //*Verificar que dataset de entrenamiento tiene labels de clase por cada muestra */
    if training_dataset.rows.len() != training_dataset.class_labels.len(){
        return Err(format!(
            "Training dataset has {} samples but {} class labels",
            training_dataset.rows.len(), training_dataset.class_labels.len()
        ));
    }

    let mut all_nearest_neighbors = Vec::new();
    

    //*Por cada muestra del dataset objetivo o de testeo */
    for(i,target) in target_dataset.rows.iter().enumerate(){
        //Calcula la distancia a todas las muestras de entrenamiento

        let mut distancias: Vec<(usize,f64,String)> = training_dataset.rows
        .iter()
        .enumerate()
        .map(|(j,row)|{
            let distancia = euclidean_distance(target, row);
            let class = training_dataset.class_labels[j].clone();
            (j,distancia,class)
        }).collect();
    




        //Sorting 
        distancias.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());

        if !distancias.is_empty() {
            let target_features = target.iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<String>>()
            .join(", ");
            
        // Mostrar los datos de la muestra más cercana
        let closest_sample = training_dataset.rows[distancias[0].0].iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<String>>()
            .join(", ");


            println!(
                "Test {} [{}] más cercano: Sample número: {} [{}] con distancia {:.4}, Clase {}",
                i, 
                target_features,
                distancias[0].0, 
                closest_sample,
                distancias[0].1, 
                distancias[0].2
            );
        }

        //Truncar al neighbor mas cercano si uno existe
        let k_actual = std::cmp::min(k, distancias.len());
        let nearest_neighbors = distancias[0..k_actual].to_vec();

        all_nearest_neighbors.push(nearest_neighbors);
  
    }

    
    Ok(all_nearest_neighbors)
}


fn find_k_nearest(
    dataset: &Dataset,target: &[f64],k:usize
) -> Result<Vec<(usize,f64,String)>,String>{

    if k == 0{
        return Err("K debe ser mayor a 0".to_string());
    }
    if dataset.rows.is_empty(){
        return Err("Dataset esta vacio".to_string());
    }
    //Calcular distancias
    let mut distancias: Vec<(usize,f64,String)> = dataset
        .rows
        .iter()
        .enumerate()
        .map(|(j,row)|{
            let distancia = euclidean_distance(&target, &row);
            let clase = dataset.class_labels[j].clone();
            (j,distancia,clase)
        }).collect();
        //Sort por mas cercano 
        distancias.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());


        //Truncar al k mas cercano
        distancias.truncate(k);


        Ok(distancias)
    
}
// Function to predict the class for a single sample based on its nearest neighbors
pub fn predict_class(neighbors: &[(usize, f64, String)]) -> String {
    // Count occurrences of each class
    let mut class_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    
    for (_, _, class) in neighbors {
        *class_counts.entry(class.clone()).or_insert(0) += 1;
    }
    
    // Find the most common class
    class_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(class, _)| class)
        .unwrap_or_else(|| "Unknown".to_string())
}

// Function to evaluate KNN model on the target dataset
pub fn evaluate_knn(
    target_dataset: &Dataset,
    training_dataset: &Dataset,
    k: usize
) -> Result<(f64, Vec<String>), String> {
    let neighbors_result = matrix_euclidean(target_dataset, training_dataset, k)?;
    
    // Only perform evaluation if target dataset has class labels
    if target_dataset.class_labels.len() != target_dataset.rows.len() {
        return Err("Cannot evaluate: target dataset lacks class labels".to_string());
    }
    
    let mut predictions = Vec::new();
    let mut correct_count = 0;
    
    for (i, neighbors) in neighbors_result.iter().enumerate() {
        let predicted_class = predict_class(neighbors);
        predictions.push(predicted_class.clone());
        
        if predicted_class == target_dataset.class_labels[i] {
            correct_count += 1;
        }
    }
    
    let accuracy = correct_count as f64 / target_dataset.rows.len() as f64;
    
    Ok((accuracy, predictions))
}