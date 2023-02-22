use ndarray::{arr1, arr2};
use nnfsrs::example_data::{
    BIASES_1_3, INPUTS_1_4, INPUTS_1_4_B, INPUTS_1_4_C, WEIGHTS_3_4,
};
use nnfsrs::p3_dot_products::{loops, ndarray_dot, with_dot, zip_loops};
use nnfsrs::p4_batches_layers_objects::batch_dot;

fn main() {
    // Basic neuron with hard-coded values
    let inputs = INPUTS_1_4;

    let batch_inputs = [
        inputs, // same
        INPUTS_1_4_B,
        INPUTS_1_4_C,
    ];
    let nd_inputs = arr2(&batch_inputs);

    let weights = WEIGHTS_3_4;
    // let weights2 = WEIGHTS_3_3;
    // let biases2 = BIASES_1_3B;

    let nd_weights = arr2(&weights);

    let biases = BIASES_1_3;
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
