use rand::Rng;
use super::distancias::euclidean_distance;




/*

Kmeans ++ escoge el primer centroide aleatoriamente 
y los demas son los mas alejados del primero 

*/
pub fn kmeans_plus_plus(data: &Vec<Vec<f64>>, k: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut centroids = Vec::new();
    
    // Elige el primer centroide al azar
    centroids.push(data[rng.gen_range(0..data.len())].clone());
    
    while centroids.len() < k {
        let mut distances: Vec<f64> = data.iter()
            .map(|point| centroids.iter()
                 .map(|c| euclidean_distance(point, c))
                 .fold(f64::INFINITY, f64::min))
            .collect();
        
        // Probabilidad proporcional a la distancia al centroide más cercano
        let total_distance: f64 = distances.iter().sum();
        let mut cumulative_prob = 0.0;
        let target = rng.gen::<f64>() * total_distance;

        for (i, &d) in distances.iter().enumerate() {
            cumulative_prob += d;
            if cumulative_prob >= target {
                centroids.push(data[i].clone());
                break;
            }
        }
    }

    println!("Centroides seleccionados:{:?}",centroids);
    
    centroids
}