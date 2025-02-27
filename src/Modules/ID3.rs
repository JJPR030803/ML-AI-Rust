use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use crate::Modules::create_dataset::DatosCualitativos;

// Node structure for the decision tree
#[derive(Debug, Clone)]
pub enum DecisionNode {
    Leaf {
        class: String,
        count: usize,
    },
    Internal {
        attribute: String,
        branches: HashMap<String, Box<DecisionNode>>,
        default_class: String,
    },
}

// Calculate entropy using Shannon's formula: -sum(p_i * log2(p_i))
pub fn calculate_entropy<T: AsRef<str> + Eq + Hash>(classes: &[T]) -> f64 {
    if classes.is_empty() {
        return 0.0;
    }
    
    let total = classes.len() as f64;
    let mut counts = HashMap::new();
    
    // Count occurrences of each class
    for class in classes {
        *counts.entry(class.as_ref()).or_insert(0) += 1;
    }
    
    // Calculate entropy
    counts.values()
        .map(|&count| {
            let probability = count as f64 / total;
            -probability * probability.log2()
        })
        .sum()
}

// Calculate information gain for a specific attribute
pub fn calculate_information_gain(
    data: &[DatosCualitativos],
    target_attr: &str,
    attribute: &str,
) -> f64 {
    // Extract the target classes
    let target_values: Vec<String> = data.iter()
        .map(|d| get_attribute_value(d, target_attr))
        .collect();
    
    // Calculate base entropy
    let base_entropy = calculate_entropy(&target_values);
    
    // Group data by attribute values
    let mut attribute_groups: HashMap<String, Vec<String>> = HashMap::new();
    for item in data {
        let attr_value = get_attribute_value(item, attribute);
        let target_value = get_attribute_value(item, target_attr);
        attribute_groups.entry(attr_value).or_insert_with(Vec::new).push(target_value);
    }
    
    // Calculate weighted entropy sum
    let total = target_values.len() as f64;
    let weighted_entropy = attribute_groups.values()
        .map(|group| {
            let weight = group.len() as f64 / total;
            weight * calculate_entropy(group)
        })
        .sum::<f64>();
    
    // Information gain = base entropy - weighted entropy
    base_entropy - weighted_entropy
}

// Helper function to get attribute value from DatosCualitativos
fn get_attribute_value(item: &DatosCualitativos, attribute: &str) -> String {
    match attribute {
        "negocio" => item.negocio.clone(),
        "calificacion" => item.calificacion.clone(),
        "estilo" => item.estilo.clone(),
        "recomendado" => item.recomendado.clone(),
        "fecha_resena" => item.fecha_resena.clone(),
        "tipo_comida" => item.tipo_comida.clone(),
        _ => panic!("Attribute '{}' not found in DatosCualitativos", attribute),
    }
}

// Build a decision tree using ID3 algorithm
pub fn build_decision_tree(
    data: &[DatosCualitativos],
    attributes: &[&str],
    target_attr: &str,
    min_samples: usize,
) -> DecisionNode {
    // If dataset is empty, return default leaf
    if data.is_empty() {
        return DecisionNode::Leaf {
            class: "unknown".to_string(),
            count: 0,
        };
    }
    
    // Extract target classes and their counts
    let mut class_counts: HashMap<String, usize> = HashMap::new();
    for item in data {
        let class = get_attribute_value(item, target_attr);
        *class_counts.entry(class).or_insert(0) += 1;
    }
    
    // Get majority class
    let (majority_class, majority_count) = class_counts.iter()
        .max_by_key(|(_, &count)| count)
        .map(|(class, &count)| (class.clone(), count))
        .unwrap_or(("unknown".to_string(), 0));
    
    // If all examples have the same class or remaining data is too small
    if class_counts.len() == 1 || data.len() <= min_samples || attributes.is_empty() {
        return DecisionNode::Leaf {
            class: majority_class,
            count: majority_count,
        };
    }
    
    // Find best attribute to split on using information gain
    let best_attribute = attributes.iter()
        .max_by(|&a, &b| {
            let gain_a = calculate_information_gain(data, target_attr, a);
            let gain_b = calculate_information_gain(data, target_attr, b);
            gain_a.partial_cmp(&gain_b).unwrap()
        })
        .copied()
        .unwrap();
    
    // Create remaining attributes list (without the best one)
    let remaining_attributes: Vec<&str> = attributes.iter()
        .filter(|&&attr| attr != best_attribute)
        .copied()
        .collect();
    
    // Group data by best attribute values
    let mut value_groups: HashMap<String, Vec<DatosCualitativos>> = HashMap::new();
    for item in data {
        let value = get_attribute_value(item, best_attribute);
        value_groups.entry(value).or_insert_with(Vec::new).push(item.clone());
    }
    
    // Create branches for each value
    let mut branches = HashMap::new();
    for (value, group) in value_groups {
        let subtree = build_decision_tree(&group, &remaining_attributes, target_attr, min_samples);
        branches.insert(value, Box::new(subtree));
    }
    
    DecisionNode::Internal {
        attribute: best_attribute.to_string(),
        branches,
        default_class: majority_class,
    }
}

// Predict class for a single data point using the decision tree
pub fn predict(tree: &DecisionNode, item: &DatosCualitativos) -> String {
    match tree {
        DecisionNode::Leaf { class, .. } => class.clone(),
        DecisionNode::Internal { attribute, branches, default_class } => {
            let value = get_attribute_value(item, attribute);
            match branches.get(&value) {
                Some(subtree) => predict(subtree, item),
                None => default_class.clone(), // Use default if value not in training data
            }
        }
    }
}