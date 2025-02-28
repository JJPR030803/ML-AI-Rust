use plotters::prelude::*;
use plotters::style::RGBColor;

// Define a color palette for different clusters
fn get_cluster_color(cluster_id: usize) -> RGBColor {
    match cluster_id % 10 {
        0 => RGBColor(31, 119, 180),   // blue
        1 => RGBColor(255, 127, 14),   // orange
        2 => RGBColor(44, 160, 44),    // green
        3 => RGBColor(214, 39, 40),    // red
        4 => RGBColor(148, 103, 189),  // purple
        5 => RGBColor(140, 86, 75),    // brown
        6 => RGBColor(227, 119, 194),  // pink
        7 => RGBColor(127, 127, 127),  // gray
        8 => RGBColor(188, 189, 34),   // olive
        9 => RGBColor(23, 190, 207),   // cyan
        _ => RGBColor(0, 0, 0),        // black (fallback)
    }
}

pub fn plot_centroides(
    data: Vec<Vec<f64>>, 
    centroids: &Vec<Vec<f64>>, 
    min: f64, 
    max: f64, 
    filepath: &str,
    cluster_assignments: Option<&Vec<usize>>,
    show_classes: bool,
    title: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filepath, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let caption = title.unwrap_or("K-Means++ Clusters");
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("Arial", 20))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min..max, min..max)?;

    chart.configure_mesh()
        .x_desc("Feature 1 (Height)")
        .y_desc("Feature 2 (Weight)")
        .draw()?;

    // If we have cluster assignments, use them to color the points
    if let Some(assignments) = cluster_assignments {
        // Group data points by cluster for better legend
        for cluster_id in 0..centroids.len() {
            let cluster_color = get_cluster_color(cluster_id);
            
            // Draw only points belonging to this cluster
            let cluster_points: Vec<_> = data.iter().zip(assignments.iter())
                .filter(|(_, &c)| c == cluster_id)
                .map(|(point, _)| {
                    if point.len() >= 2 {
                        (point[0], point[1])
                    } else {
                        (0.0, 0.0)
                    }
                })
                .collect();
                
            if !cluster_points.is_empty() {
                chart.draw_series(
                    cluster_points.iter().map(|&(x, y)| {
                        Circle::new((x, y), 5, cluster_color.filled())
                    })
                )?
                .label(format!("Cluster {}", cluster_id))
                .legend(move |(x, y)| Circle::new((x, y), 5, cluster_color.filled()));
            }
        }
    } else {
        // No cluster assignments, just draw all points in blue
        chart.draw_series(
            data.iter().map(|dato| {
                if dato.len() >= 2 {
                    Circle::new((dato[0], dato[1]), 5, BLUE.filled())
                } else {
                    Circle::new((0.0, 0.0), 5, BLUE.filled())
                }
            }),
        )?
        .label("Data Points")
        .legend(|(x, y)| Circle::new((x, y), 5, BLUE.filled()));
    }

    // Graficar centroides
    chart.draw_series(
        centroids.iter().enumerate().map(|(i, centroid)| {
            if centroid.len() >= 2 {
                // Fixed: Removed PointStyle and simplified the element creation
                let size = 10;
                EmptyElement::at((centroid[0], centroid[1]))
                    + Cross::new((0, 0), size, Into::<ShapeStyle>::into(&BLACK).stroke_width(2))
                    + Circle::new((0, 0), size, get_cluster_color(i).filled())
            } else {
                // Match the return type with the if branch
                EmptyElement::at((0.0, 0.0))
                    + Cross::new((0, 0), 0, BLACK.stroke_width(0))
                    + Circle::new((0, 0), 0, BLACK.filled())
            }
        }),
    )?
    .label("Centroids")
    .legend(|(x, y)| Cross::new((x, y), 10, BLACK.stroke_width(2)));

    // Añadir leyenda
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Plot 3D data points and centroids with cluster coloring
pub fn plot_3d(
    data: &Vec<Vec<f64>>, 
    centroids: &Vec<Vec<f64>>,
    min: f64,
    max: f64,
    filepath: &str,
    feature_names: &[String],
    cluster_assignments: Option<&Vec<usize>>,
    show_classes: bool,
    title: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Make sure we have 3 feature names, or use defaults
    let x_label = feature_names.get(0).map_or("Feature 1", |s| s.as_str());
    let y_label = feature_names.get(1).map_or("Feature 2", |s| s.as_str());
    let z_label = feature_names.get(2).map_or("Feature 3", |s| s.as_str());

    // Create a drawing area
    let root = BitMapBackend::new(filepath, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Split into 2 panels: top-down view and 3D view
    let (top_panel, bottom_panel) = root.split_vertically(384);
    
    // Further split bottom panel for different angle views
    let (bottom_left, bottom_right) = bottom_panel.split_horizontally(512);

    // Get plot title
    let plot_title = title.unwrap_or("K-Means++ Clusters 3D View");

    // Top-down view (X-Y plane)
    let mut top_chart = ChartBuilder::on(&top_panel)
        .caption(format!("{}: {} vs {} (Top View)", plot_title, x_label, y_label), ("Arial", 20))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(min..max, min..max)?;

    top_chart.configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    // Draw data points on top-down view with cluster coloring
    if let Some(assignments) = cluster_assignments {
        for cluster_id in 0..centroids.len() {
            let cluster_color = get_cluster_color(cluster_id);
            
            let filtered_data: Vec<_> = data.iter()
                .zip(assignments.iter())
                .filter(|(point, &c)| c == cluster_id && point.len() >= 3)
                .map(|(point, _)| (point[0], point[1]))
                .collect();
                
            if !filtered_data.is_empty() {
                top_chart.draw_series(
                    filtered_data.iter().map(|&(x, y)| {
                        Circle::new((x, y), 3, cluster_color.filled())
                    })
                )?;
            }
        }
    } else {
        // No cluster assignments, draw all points in blue
        top_chart.draw_series(
            data.iter()
                .filter(|point| point.len() >= 3)
                .map(|point| Circle::new((point[0], point[1]), 3, BLUE.filled()))
        )?;
    }

    // Draw centroids on top-down view
    top_chart.draw_series(
        centroids.iter().enumerate()
            .filter(|(_, centroid)| centroid.len() >= 3)
            .map(|(i, centroid)| {
                let center = (centroid[0], centroid[1]);
                EmptyElement::at(center)
                    + Cross::new((0, 0), 8, Into::<ShapeStyle>::into(&BLACK).stroke_width(2))
                    + Circle::new((0, 0), 8, get_cluster_color(i).filled())
            })
    )?;

    // Create 3D-like view using X-Z plane (front view)
    let mut front_chart = ChartBuilder::on(&bottom_left)
        .caption(format!("{}: {} vs {} (Front View)", plot_title, x_label, z_label), ("Arial", 20))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(min..max, min..max)?;

    front_chart.configure_mesh()
        .x_desc(x_label)
        .y_desc(z_label)
        .draw()?;

    // Draw data points on front view with cluster coloring
    if let Some(assignments) = cluster_assignments {
        for cluster_id in 0..centroids.len() {
            let cluster_color = get_cluster_color(cluster_id);
            
            let filtered_data: Vec<_> = data.iter()
                .zip(assignments.iter())
                .filter(|(point, &c)| c == cluster_id && point.len() >= 3)
                .map(|(point, _)| (point[0], point[2]))
                .collect();
                
            if !filtered_data.is_empty() {
                front_chart.draw_series(
                    filtered_data.iter().map(|&(x, y)| {
                        Circle::new((x, y), 3, cluster_color.filled())
                    })
                )?;
            }
        }
    } else {
        // No cluster assignments, draw all in blue
        front_chart.draw_series(
            data.iter()
                .filter(|point| point.len() >= 3)
                .map(|point| Circle::new((point[0], point[2]), 3, BLUE.filled()))
        )?;
    }

    // Draw centroids on front view
    front_chart.draw_series(
        centroids.iter().enumerate()
            .filter(|(_, centroid)| centroid.len() >= 3)
            .map(|(i, centroid)| {
                let center = (centroid[0], centroid[2]);
                EmptyElement::at(center)
                    + Cross::new((0, 0), 8, Into::<ShapeStyle>::into(&BLACK).stroke_width(2))
                    + Circle::new((0, 0), 8, get_cluster_color(i).filled())
            })
    )?;

    // Create 3D-like view using Y-Z plane (side view)
    let mut side_chart = ChartBuilder::on(&bottom_right)
        .caption(format!("{}: {} vs {} (Side View)", plot_title, y_label, z_label), ("Arial", 20))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(min..max, min..max)?;

    side_chart.configure_mesh()
        .x_desc(y_label)
        .y_desc(z_label)
        .draw()?;

    // Draw data points on side view with cluster coloring
    if let Some(assignments) = cluster_assignments {
        for cluster_id in 0..centroids.len() {
            let cluster_color = get_cluster_color(cluster_id);
            
            let filtered_data: Vec<_> = data.iter()
                .zip(assignments.iter())
                .filter(|(point, &c)| c == cluster_id && point.len() >= 3)
                .map(|(point, _)| (point[1], point[2]))
                .collect();
                
            if !filtered_data.is_empty() {
                side_chart.draw_series(
                    filtered_data.iter().map(|&(x, y)| {
                        Circle::new((x, y), 3, cluster_color.filled())
                    })
                )?;
            }
        }
    } else {
        // No cluster assignments
        side_chart.draw_series(
            data.iter()
                .filter(|point| point.len() >= 3)
                .map(|point| Circle::new((point[1], point[2]), 3, BLUE.filled()))
        )?;
    }

    // Draw centroids on side view
    side_chart.draw_series(
        centroids.iter().enumerate()
            .filter(|(_, centroid)| centroid.len() >= 3)
            .map(|(i, centroid)| {
                let center = (centroid[1], centroid[2]);
                EmptyElement::at(center)
                    + Cross::new((0, 0), 8, Into::<ShapeStyle>::into(&BLACK).stroke_width(2))
                    + Circle::new((0, 0), 8, get_cluster_color(i).filled())
            })
    )?;

    // Add color legend
    let legend_area = bottom_right.clone().margin(5, 5, 5, 5);
    
    // Draw cluster legend
    let mut legend_y = 40;
    
    // Fixed: Use a proper TextStyle
    let text_style = ("Arial", 15).into_text_style(&legend_area);
    // Draw legend title
    legend_area.draw_text(
        "Legend:",
        &text_style,
        (700, legend_y)
    )?;
    legend_y += 25;
    
    // Draw centroid legend entry
   // Draw centroid legend entry
legend_area.draw(&(EmptyElement::at((700, legend_y)) 
+ Cross::new((0, 0), 8, BLACK.stroke_width(2))
+ Circle::new((0, 0), 8, RED.filled())))?;

legend_area.draw_text(
"Centroid",
&("Arial", 12).into_text_style(&legend_area),
(720, legend_y)
)?;
    
    
    legend_y += 20;
    
    // Draw cluster legend entries
    if let Some(_) = cluster_assignments {
        for i in 0..centroids.len() {
            let color = get_cluster_color(i);
            legend_area.draw(&Circle::new((700, legend_y), 5, color.filled()))?;
            legend_area.draw_text(
                &format!("Cluster {}", i),
                &("Arial", 12).into_text_style(&legend_area),
                (720, legend_y)
            )?;
            legend_y += 20;
        }
    } else {
        legend_area.draw(&Circle::new((700, legend_y), 5, BLUE.filled()))?;
        legend_area.draw_text(
            "Data Points",
            &("Arial", 12).into_text_style(&legend_area),
            (720, legend_y)
        )?;
    }

    root.present()?;
    Ok(())
}