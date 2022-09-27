fn main() {
    // Basic neuron with hard-coded values
    let inputs = [1.0, 2.0, 3.0, 2.5];
    let weights = [0.2, 0.8, -0.5, 1.0];
    let bias = 2.0;
    let output = 
        inputs[0] * weights[0] +
        inputs[1] * weights[1] +
        inputs[2] * weights[2] +
        inputs[3] * weights[3] +
        bias;
    println!("output: {}", output);
}
