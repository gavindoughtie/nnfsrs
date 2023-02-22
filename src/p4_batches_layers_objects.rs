use crate::p3_dot_products::dot;
use ndarray::{arr1, arr2};

pub fn transpose(inputs: &[[f64; 4]; 3]) -> [[f64; 3]; 4] {
    let mut transposed = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    transposed[0] = [inputs[0][0], inputs[1][0], inputs[2][0]];
    transposed[1] = [inputs[0][1], inputs[1][1], inputs[2][1]];
    transposed[2] = [inputs[0][2], inputs[1][2], inputs[2][2]];
    transposed[3] = [inputs[0][3], inputs[1][3], inputs[2][3]];

    return transposed;
}

pub fn dot_product(inputs: &[[f64; 4]; 3], weights: &[[f64; 4]; 3]) -> [[f64; 4]; 3] {
    let mut outputs = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    outputs[0][0] = dot(&inputs[0], &weights[0]);
    outputs[0][1] = dot(&inputs[0], &weights[1]);
    outputs[0][2] = dot(&inputs[0], &weights[2]);
    outputs[0][3] = dot(&inputs[0], &weights[3]);
    outputs[1][0] = dot(&inputs[1], &weights[0]);
    outputs[1][1] = dot(&inputs[1], &weights[1]);
    outputs[1][2] = dot(&inputs[1], &weights[2]);
    outputs[1][3] = dot(&inputs[1], &weights[3]);
    outputs[2][0] = dot(&inputs[2], &weights[0]);
    outputs[2][1] = dot(&inputs[2], &weights[1]);
    outputs[2][2] = dot(&inputs[2], &weights[2]);
    outputs[2][3] = dot(&inputs[2], &weights[3]);

    outputs
}

pub fn add_bias(inputs: &[[f64; 4]; 3], bias: &[f64; 4]) -> [[f64; 4]; 3] {
    let mut outputs = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    outputs[0][0] = inputs[0][0] + bias[0];
    outputs[0][1] = inputs[0][1] + bias[1];
    outputs[0][2] = inputs[0][2] + bias[2];
    outputs[0][3] = inputs[0][3] + bias[3];
    outputs[1][0] = inputs[1][0] + bias[0];
    outputs[1][1] = inputs[1][1] + bias[1];
    outputs[1][2] = inputs[1][2] + bias[2];
    outputs[1][3] = inputs[1][3] + bias[3];
    outputs[2][0] = inputs[2][0] + bias[0];
    outputs[2][1] = inputs[2][1] + bias[1];
    outputs[2][2] = inputs[2][2] + bias[2];
    outputs[2][3] = inputs[2][3] + bias[3];

    outputs
}

pub fn f_mult(input_vector: &[f64], weights_vector: &[f64; 3]) -> f64 {
    let dot_result = dot(input_vector, weights_vector);
    println!(
        "input_vector: {:#?}\nweights_vector: {:#?}\ndot_result: {:#?}",
        input_vector, weights_vector, dot_result
    );
    return dot_result;
}

pub fn batch_dot(inputs: &[[f64; 4]; 3], weights: &[[f64; 4]; 3], biases: &[f64]) -> [[f64; 3]; 3] {
    let nd_inputs = arr2(inputs);
    let nd_weights = arr2(weights);
    let nd_biases = arr1(biases);
    let nd_transposed = nd_weights.t();
    let nd_result = nd_inputs.dot(&nd_transposed) + nd_biases.clone();
    println!(r#"nd_inputs shape: {:#?}\n
       nd_weights shape {:#?}\n
       nd_transposed shape {:#?}\n
       nd_result shape {:#?}"#,
      nd_inputs.shape(), nd_weights.shape(), nd_transposed.shape(), nd_result.shape());
    // println!("nd_inputs:\n{:#?}\nnd_weights:\n{:#?}\nnd_transposed:\n{:#?}\nnd_biases:\n{:#?}\nnd_result:\n{:#?}", nd_inputs, nd_weights, nd_transposed, nd_biases, nd_result);
    assert_eq!(inputs.len(), weights.len());
    let transposed_weights = transpose(weights);
    println!(
        "input: {:#?}, weights: {:#?}, transposed weights: {:#?}",
        inputs, weights, transposed_weights
    );
    let mut result = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ];
    for i in 0..inputs.len() {
      for j in 0..weights.len() {
        result[i][j] += (inputs[i][j] * weights[i][j]) + biases[i];
      }
    }
    // let result: Vec<_> = inputs
    //     .iter()
    //     .map(|input_vector| transposed_weights.iter()
    //       .map(|weight_vector| f_mult(input_vector, &weight_vector))
    //       .map(|mw| biases.clone().iter().map(move |b| mw + b))
    //     )
    //     .collect();
    // println!("batch dot result: {:#?}, biases: {:#?}", result, biases);
    // let faux_result = [
    //   [0.0, 0.0, 0.0],
    //   [0.0, 0.0, 0.0],
    //   [0.0, 0.0, 0.0],
    // ];
    return result;
}
