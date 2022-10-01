use ndarray::{arr1, arr2};
use nnfsrs::p3_dot_products::{loops, ndarray_dot, with_dot, zip_loops};
use nnfsrs::p4_batches_layers_objects::batch_dot;

fn main() {
    // Basic neuron with hard-coded values
    let inputs = [1.0, 2.0, 3.0, 2.5];

    let batch_inputs = [
        inputs, // same
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ];
    let nd_inputs = arr2(&batch_inputs);

    let weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];
    let weights2 = [
      [
        0.1, -0.14, 0.5
      ],
      [
        -0.5, 0.12, -0.33
      ],
      [
        -0.44, 0.73, -0.13
      ]
    ];
    let biases2 = [-1.0, 2.0, -0.5];

    let nd_weights = arr2(&weights);

    let biases = [2.0, 3.0, 0.5];
    let nd_biases = arr1(&biases);

    let batch_dot_results = batch_dot(&batch_inputs, &weights, &biases);
    let nd_dot_results = nd_inputs.dot(&nd_weights.t());
    println!(
        "nd_dot_results shape: {:#?}, nd_biases shape: {:#?}",
        nd_dot_results.shape(),
        nd_biases.shape()
    );
    println!("batch_dot_results: {:#?}", batch_dot_results);
    println!("batch_dot_results shape: {:#?}", batch_dot_results);

    println!("ndarray.dot:");
    ndarray_dot();
    println!("processing with loops:");
    loops();
    println!("processing with idiomatic map/zip loops (should be same as above):");
    zip_loops();
    println!("processing with a dot product function");
    with_dot(&inputs, &weights, &biases);
}
