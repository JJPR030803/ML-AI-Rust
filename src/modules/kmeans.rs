use rand::Rng;
use super::distancias::euclidean_distance;

/*
Kmeans ++ escoge el primer centroide aleatoriamente 
y los demas son los mas alejados del primero 
*/
pub fn kmeans_plus_plus(data: &Vec<Vec<f64>>, k: usize) -> Vec<Vec<f64>> {
    if data.is_empty() || k == 0 || k > data.len() {
        return Vec::new();
    }

    let mut rng = rand::thread_rng();
    let mut centroids = Vec::with_capacity(k);
    
    // Elige el primer centroide al azar
    centroids.push(data[rng.gen_range(0..data.len())].clone());
    
    while centroids.len() < k {
        let mut distances: Vec<f64> = data.iter()
            .map(|point| {
                centroids.iter()
                    .map(|c| euclidean_distance(point, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        
        // Evitar división por cero
        let total_distance: f64 = distances.iter().sum();
        if total_distance <= 0.0 {
            // Si las distancias son todas 0, elegir al azar
            centroids.push(data[rng.gen_range(0..data.len())].clone());
            continue;
        }

        // Probabilidad proporcional a la distancia al centroide más cercano
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

        // Seguro adicional para evitar bucles infinitos
        if !selected {
            centroids.push(data[rng.gen_range(0..data.len())].clone());
        }
    }

    println!("Centroides iniciales seleccionados: {} centroides", centroids.len());
    
    centroids
}

/*  
Asigna a un cluster cada punto del dato usando distancia euclidiana
si son 3 k o centroides seria (0,1,2) los posibles valores de salida
*/
pub fn asignar_a_clusters(data: Vec<Vec<f64>>, centroides: Vec<Vec<f64>>) -> Vec<usize> {
    if data.is_empty() || centroides.is_empty() {
        return Vec::new();
    }

    data
        .iter()
        .map(|point| {
            centroides
                .iter()
                .enumerate()
                .map(|(i, centroid)| (i, euclidean_distance(point, centroid)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0) // Failsafe
        })
        .collect()
}

/*
Calculamos la media de los datos dentro de cada centrooide asignado 
y actualizamos el punto medio del centroide a que sea la media obtenida
*/
pub fn actualizar_centroides(data: &Vec<Vec<f64>>, clusters: &Vec<usize>, k: usize) -> Vec<Vec<f64>> {
    if data.is_empty() || clusters.is_empty() || data.len() != clusters.len() {
        return vec![vec![0.0; data.get(0).map_or(0, |v| v.len())]; k];
    }

    let dim = data[0].len();
    let mut nuevos_centroides = vec![vec![0.0; dim]; k];
    let mut counts = vec![0; k];

    for (point, &cluster) in data.iter().zip(clusters.iter()) {
        if cluster < k {  // Validar que el cluster es válido
            for (i, &value) in point.iter().enumerate() {
                if i < dim {  // Validar que el índice es válido
                    nuevos_centroides[cluster][i] += value;
                }
            }
            counts[cluster] += 1;
        }
    }

    for (i, centroid) in nuevos_centroides.iter_mut().enumerate() {
        if counts[i] > 0 {
            for value in centroid.iter_mut() {
                *value /= counts[i] as f64;
            }
        } else {
            // Si un centroide no tiene puntos asignados, mantenerlo en su posición actual
            // o asignarle un punto aleatorio
            println!("Advertencia: Centroide {} sin puntos asignados", i);
        }
    }
    
    nuevos_centroides
}

pub fn iterar_hasta_optimo(max_iters: usize, data: Vec<Vec<f64>>, k: usize) -> Vec<Vec<f64>> {
    if data.is_empty() || k == 0 || k > data.len() {
        return Vec::new();
    }
    
    let mut centroids = kmeans_plus_plus(&data, k);
    let mut iteration = 0;
    
    while iteration < max_iters {
        let clusters = asignar_a_clusters(data.clone(), centroids.clone());
        let new_centroids = actualizar_centroides(&data, &clusters, k);

        // Verificar convergencia
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

        if converged {
            println!("KMeans convergió en {} iteraciones", iteration + 1);
            break;
        }
        
        centroids = new_centroids;
        iteration += 1;
    }
    
    if iteration == max_iters {
        println!("KMeans alcanzó el máximo de iteraciones ({}) sin converger", max_iters);
    }
    
    centroids
}