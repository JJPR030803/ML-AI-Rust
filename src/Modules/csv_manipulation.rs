use std::string;

use csv::Writer;
use nalgebra::DMatrix;

use super::create_dataset::DatosCualitativos;



pub fn write_matrix_to_csv(filename: &str, matrix: &DMatrix<f64>) -> Result<(),Box<dyn std::error::Error>>{
let mut wtr = Writer::from_path(filename)?;

for row in 0..matrix.nrows(){
    let row_data: Vec<f64> = (0..matrix.ncols()).map(|col|matrix[(row,col)]).collect();
    wtr.serialize(row_data)?;
}


wtr.flush()?;
println!("Datos escritos a: {}",filename);

Ok(())
}

pub fn write_vec_tocsv(filename: &str, data: &[DatosCualitativos]) -> Result<(), Box<dyn std::error::Error>>{
    let mut wtr = Writer::from_path(filename)?;
    for record in data{
        wtr.serialize(record)?;
    }

    wtr.flush()?;

    println!("Datos escritos a {}",filename);

    Ok(())

}