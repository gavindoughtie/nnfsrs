pub fn dot_product(inputs: &[[f64; 4]; 3], weights: &[[f64; 4]; 3]) -> [[f64; 4]; 3] {
    let mut outputs = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    for i in 0..3 {
        for j in 0..4 {
            outputs[i][j] = dot(&inputs[i], &weights[j]);
        }
    }
    outputs
}

pub fn add_bias(inputs: &[[f64; 4]; 3], bias: &[f64; 4]) -> [[f64; 4]; 3] {
    let mut outputs = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    for i in 0..3 {
        for j in 0..4 {
            outputs[i][j] = inputs[i][j] + bias[j];
        }
    }
    outputs
}
