use ndarray::{arr1, arr2};

pub fn ndarray_dot() {
    let inputs = arr1(&[1.0, 2.0, 3.0, 2.5]);
    let weights = arr2(&[
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]);
    let biases = arr1(&[2.0, 3.0, 0.5]);
    let dotted = weights.dot(&inputs);
    println!("dotted: {}", dotted);

    let result = weights.dot(&inputs) + biases;

    println!("{}", result);
}

pub fn loops() {
    // Basic neuron with hard-coded values
    let inputs = [1.0, 2.0, 3.0, 2.5];

    let weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];

    let biases = [2.0, 3.0, 0.5];

    let mut layer_outputs = vec![];

    for row in 0..3 {
        let neuron_weights = weights[row];
        let neuron_bias = biases[row];
        let mut neuron_output = 0.0;
        for neuron_index in 0..4 {
            let n_input = inputs[neuron_index];
            let n_weight = neuron_weights[neuron_index];
            neuron_output += n_input * n_weight;
        }
        neuron_output += neuron_bias;
        layer_outputs.push(neuron_output);
    }
    println!("layer_outputs: {:#?}", layer_outputs);
}

/*
let output = [
        inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
        inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
        inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3,
        ];
 */

pub fn zip_loops() {
    // Basic neuron with hard-coded values
    let inputs = [1.0, 2.0, 3.0, 2.5];

    let weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];

    let biases = [2.0, 3.0, 0.5];

    let results: Vec<f64> = weights
        .iter()
        .zip(biases.iter())
        .map(|(weights, bias)| {
            weights
                .iter()
                .zip(inputs.iter())
                .map(|(w, i)| w * i)
                .sum::<f64>()
                + bias
        })
        .collect();

    println!("{:#?}", results);
}

pub fn dot(arr1: &[f64], arr2: &[f64]) -> f64 {
    assert_eq!(arr1.len(), arr2.len());
    arr1.iter().zip(arr2.iter()).map(|(a, b)| a * b).sum()
}

pub fn add_arrays(arr1: &[f64], arr2: &[f64]) -> Vec<f64> {
    arr1.iter().zip(arr2.iter()).map(|(a, b)| a + b).collect()
}

pub fn element_add(arr1: &[f64], value: f64) -> Vec<f64> {
    arr1.iter().map(|el| el + value).collect()
}

pub fn with_dot(inputs: &[f64], weights: &[[f64; 4]], biases: &[f64]) {

    let results: Vec<_> = weights
        .iter()
        .map(|w| dot(w, &inputs))
        .zip(biases.iter())
        .map(|(d, b)| d + b)
        .collect();

    println!("{:#?}", results);

    let a1 = [3.0, 6.0, 9.0];
    let a2 = [1.0, 2.0, 3.0];
    let a1_nd = arr1(&a1);
    let a2_nd = arr1(&a2);
    let nd_dot = a1_nd.dot(&a2_nd);
    let my_dot = dot(&a1, &a2);
    println!("nd_dot: {}, my_dot: {}", nd_dot, my_dot);
}
