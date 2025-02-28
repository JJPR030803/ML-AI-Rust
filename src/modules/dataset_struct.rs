

pub struct Dataset{
    pub rows: Vec<Vec<f64>>,
    pub class_labels: Vec<String>,
    pub feature_names:Vec<String>
}



impl Dataset{
    pub fn new(rows:Vec<Vec<f64>>,feature_names:Vec<String>,class_labels:Vec<String>)->Self{
        Dataset{
            rows,
            feature_names,
            class_labels,
        }
    }
    pub fn display(&self) {
        println!("Features: {:?}", self.feature_names);
        println!("Samples:");
        
        for (i, row) in self.rows.iter().enumerate() {
            let label = if i < self.class_labels.len() {
                &self.class_labels[i]
            } else {
                "Unknown"
            };
            println!("{}: {:?} - Class: {}", i, row, label);
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        if self.rows.is_empty() {
            return (0, self.feature_names.len());
        }
        (self.rows.len(), self.rows[0].len())
    }
}
