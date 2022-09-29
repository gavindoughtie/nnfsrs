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

pub fn zip_loops() {
    // Basic neuron with hard-coded values
    let inputs = [1.0, 2.0, 3.0, 2.5];

    let weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];

    let biases = [2.0, 3.0, 0.5];

    // let zipped: Vec<((&f64, &[f64; 4]), &f64)> =
    let results: Vec<_> = inputs
        .iter()
        .zip(weights.iter())
        .zip(biases.iter())
        .map(|((input, weights), bias)| {
            println!("multiplying {} with each element in {:#?} and adding {}", input, weights, bias);
            let collected_weights: f64 = weights
                .iter()
                .map(|w| {
                    let multiplied_weight = w * input;
                    // print!("{} * {} = {}\n", w, input, multiplied_weight);
                    return multiplied_weight;
                })
                .sum();
            return collected_weights + bias;
        })
        .collect();
    // let from_iter: Vec<f64> = dummy.iter().map(|f| *f).collect();
    // let zipped: Vec<f64> = inputs.iter().zip(weights.iter()).zip(biases.iter()).map(|((n_input, neuron_weights), n_bias)| {
    //     let result = neuron_weights.map(|weight| {
    //         return n_input * weight;
    //     }).sum() + n_bias;
    //     return result;
    // }).map(|x| *x);

    // let zipped: Vec<std::iter::Zip<f64, f64>> = biases.iter().zip(dummy.iter()).collect();

    println!("{:#?}", results);

    // let layer_outputs = inputs.iter().zip(weights.iter());
    // println!("{:#?}", layer_outputs.zip(biases.iter()).map(|((input, weights), bias)| weights.iter().map(|w| w * input + bias)));

    // let layer_outputs = inputs.iter().zip(biases.iter()).zip(weights.iter()).map(
    //     | tup | {
    //         println!("{}, {}, {:#?}", tup.0.0, tup.0.1, tup.1);
    //         // [tup.0.0, tup.0.1, tup.1]
    //         tup
    //         // |((n_input, neuron_bias), neuron_weights)| {
    //         // // | tup | {
    //         //     let val = neuron_weights.iter().map(|w| (w * n_input + neuron_bias).copy());
    //         //     return val;
    //     },
    // );

    // println!("layer_outputs: {:#?}", layer_outputs);
}
