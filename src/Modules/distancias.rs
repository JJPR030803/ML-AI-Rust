
pub fn euclidean_distance(a: &[f64], b:&[f64]) -> f64{
    if a.len() != b.len(){
        panic!("euclidean_distance: Vectores no son del mismo tamaño");
    }
    a.iter()
    .zip(b.iter())
    .map(|(x,y)| (x-y).powi(2))
    .sum::<f64>()
    .sqrt()
}