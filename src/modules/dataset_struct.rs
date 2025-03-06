use rand::Rng;

pub struct Dataset {
    pub rows: Vec<Vec<f64>>,
    pub class_labels: Vec<String>,
    pub feature_names: Vec<String>,
}

impl Dataset {
    pub fn new(rows: Vec<Vec<f64>>, feature_names: Vec<String>, class_labels: Vec<String>) -> Self {
        Dataset {
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

    /// Inicializa los datasets de testing y training
    ///
    /// # Parametros
    /// - `num_features`: Numero de clases por muestra
    /// - `num_samples`: Numero de (filas) a generar EJE X
    /// - `min` and `max`: Rango de posibles valores a generar
    /// - `feature_names`: Nombres de las clases
    /// - `possible_classes`: Etiquetas o categorias posibles para las clases
    ///
    /// # Returns
    /// Regresa una tupla (testing_dataset: Dataset, training_dataset: Dataset, feature_names: Vec<String>)
    pub fn initialize_datasets(
        num_features: usize,
        num_samples: usize,
        min: f64,
        max: f64,
        feature_names: Vec<String>,
        possible_classes: Vec<String>,
    ) -> (Dataset, Dataset, Vec<String>) {
        let mut rng = rand::thread_rng();

        // Generate random data for training
        let training_data = Self::random_f64_matrix(num_features, num_samples, min, max);

        // Generate random classes for training data
        let training_labels: Vec<String> = (0..num_samples)
            .map(|_| possible_classes[rng.gen_range(0..possible_classes.len())].clone())
            .collect();

        let training_dataset = Dataset::new(
            training_data.clone(),
            feature_names.clone(),
            training_labels,
        );

        // Generate random data for testing
        let testing_data = Self::random_f64_matrix(num_features, num_samples, min, max);

        // Generate random classes for testing data
        let testing_labels: Vec<String> = (0..num_samples)
            .map(|_| possible_classes[rng.gen_range(0..possible_classes.len())].clone())
            .collect();

        // Create testing dataset
        let testing_dataset = Dataset::new(testing_data, feature_names.clone(), testing_labels);

        (testing_dataset, training_dataset, feature_names)
    }

    // Helper function to generate a random matrix of f64 values
    fn random_f64_matrix(features: usize, samples: usize, min: f64, max: f64) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        (0..samples)
            .map(|_| (0..features).map(|_| rng.gen_range(min..max)).collect())
            .collect()
    }
}
