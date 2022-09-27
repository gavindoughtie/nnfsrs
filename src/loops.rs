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
