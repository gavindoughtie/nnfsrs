use nnfsrs::dot_products::{loops, ndarray_dot, with_dot, zip_loops};

fn main() {
    // Basic neuron with hard-coded values
    let inputs = [1.0, 2.0, 3.0, 2.5];

    let weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];

    let biases = [2.0, 3.0, 0.5];

    println!("ndarray.dot:");
    ndarray_dot();
    println!("processing with loops:");
    loops();
    println!("processing with idiomatic map/zip loops (should be same as above):");
    zip_loops();
    println!("processing with a dot product function");
    with_dot(&inputs, &weights, &biases);
}
