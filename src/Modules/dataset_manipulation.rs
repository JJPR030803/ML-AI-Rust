use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fmt::Debug;

use super::create_dataset::DatosCualitativos;

pub fn split_qualitative_dataset(
    data: Vec<DatosCualitativos>,
    test_size: f64,
    random_seed: Option<u64>,
) -> (Vec<DatosCualitativos>,Vec<DatosCualitativos>){

    if test_size < 0.0 || test_size > 1.0{
        panic!("Test size debe ser entre 0.0 y 1.0");
    }

    //Creamos una copia mutable de los datos
    let mut data_copy: Vec<DatosCualitativos> = data.clone();

    //Setup rng seed

    let mut rng = match random_seed{
        Some(seed) => ChaCha8Rng::seed_from_u64(seed),
        None => ChaCha8Rng::from_entropy(),
    };

    //shuffle datos
    data_copy.shuffle(&mut rng);


    //Calculamos el punto de split
    let split_idx = (data_copy.len() as f64 * (1.0 - test_size)) as usize;

    //Split
    let train_data = data_copy[..split_idx].to_vec();
    let test_data = data_copy[split_idx..].to_vec();

    (train_data,test_data)

}