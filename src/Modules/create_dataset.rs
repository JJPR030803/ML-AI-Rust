// Import the necessary items from nalgebra
use nalgebra::DMatrix;
use rand::{distributions::Slice,thread_rng,Rng};
use serde::Serialize;
use smartcore::linalg::basic::matrix::DenseMatrix;


#[derive(Debug, Serialize,Clone)]
pub struct DatosCualitativos{
    pub negocio: String,
    pub calificacion: String,
    pub estilo: String,
    pub recomendado: String,
    pub fecha_resena: String,
    pub tipo_comida: String

}

pub fn create_matrix_numerical(rows:usize,columnas: usize,valor_minimo:f64,valor_maximo:f64) -> DMatrix <f64>{
    let mut rng = rand::thread_rng();
    let matrix: DMatrix<f64> = DMatrix::<f64>::from_fn(rows, columnas, |_, _| rng.gen_range(valor_minimo..valor_maximo));
    matrix
}

pub fn create_qualitative_matrix(n: usize) -> Vec<DatosCualitativos>{
    let mut rng = thread_rng();
    let negocios = ["Guero Isaac","Muuz","Iceberg Lounge","Uat","Mamitas","GBD"];
    let calificaciones = ["0","1","2","3","4","5"];
    let estilos = ["Casual","Informal","Formal","No importa","N/A"];
    let recomendados = ["si","no"];
    let fechas_resenas = ["2025","2024","2023","2022","2021","2020","2019","2018"];
    let tipos_comida = ["Mexicana","Italiana","Griega","Mediterranea","China","Japonesa","Brasileña"];

    (0..n)
          .map(|_| DatosCualitativos{
            negocio: negocios[rng.gen_range(0..negocios.len())].to_string(),
            calificacion: calificaciones[rng.gen_range(0..calificaciones.len())].to_string(),
            estilo: estilos[rng.gen_range(0..estilos.len())].to_string(),
            recomendado: recomendados[rng.gen_range(0..recomendados.len())].to_string(),
            fecha_resena: fechas_resenas[rng.gen_range(0..fechas_resenas.len())].to_string(),
            tipo_comida: tipos_comida[rng.gen_range(0..tipos_comida.len())].to_string(),
            
          }).collect()

}


pub fn random_dense_matrix(rows: usize,cols:usize,min:f64,max:f64)->DenseMatrix<f64>{

    let mut rng = rand::thread_rng();

    let data: Vec<Vec<f64>> = (0..rows)
    .map(|_| (0..cols).map(|_| rng.gen_range(min..max))
    .collect()).collect();

    let refs_data: Vec<&[f64]> = data.iter().map(|row| row.as_slice()).collect();


DenseMatrix::from_2d_array(&refs_data)

}


pub fn create_lista(n:usize,min:f64,max:f64)->Vec<f64>{
    let mut rng = rand::thread_rng();
    (0..n)
    .map(|_| rng.gen_range(min..max))
    .collect()
}

pub fn random_f64_matrix(e_lista:usize,n_listas:usize,min:f64,max:f64) -> Vec<Vec<f64>> {
    (0..n_listas).map(|_| create_lista(e_lista, min, max))
    .collect()
    
}