use ndarray::array;
use ndarray_rand::rand_distr::num_traits::Float;
use rand::thread_rng;
use ndarray::{Array, Array1, Array2, ArrayBase, ArrayView, ArrayView2, Axis, Ix2, OwnedRepr, s};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::num_traits::real::Real;
use ndarray_rand::RandomExt;

fn relu(x: ArrayBase<OwnedRepr<f64>, Ix2>) -> Array2<f64> {
    x.mapv(|a| a.max(0.0))
}

fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|a| if a > 0.0 { 1.0 } else { 0.0 })
}

fn softmax(x: ArrayBase<OwnedRepr<f64>, Ix2>) -> Array2<f64> {
    let max = x.map_axis(Axis(1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
        .insert_axis(Axis(1));
    let exp_x = (&x - &max).mapv(f64::exp);
    let sum_exp_x = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));
    exp_x / &sum_exp_x
}

struct MLP {
    weight_input_hidden: Array2<f64>,
    weight_output_hidden: Array2<f64>,
    bias_hidden: Array2<f64>,
    bias_output: Array2<f64>,
    learning_rate: f64,
    hidden_input: Option<Array2<f64>>,
    hidden_output: Option<Array2<f64>>,
    final_input: Option<Array2<f64>>,
    final_output: Option<Array2<f64>>,
}

impl MLP {
    fn new_mlp(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> MLP {
        let rng = thread_rng();
        let input_weights_hidden: Array2<f64> = Array2::random((input_size, hidden_size), StandardNormal) * 0.01;
        let output_weights_hidden: Array2<f64> = Array2::random((hidden_size, output_size), StandardNormal) * 0.01;
        let bias_hidden: Array2<f64> = Array::zeros((1, hidden_size));
        let bias_output: Array2<f64> = Array::zeros((1, output_size));

        MLP {
            weight_input_hidden: input_weights_hidden,
            weight_output_hidden: output_weights_hidden,
            bias_hidden,
            bias_output,
            learning_rate,
            hidden_input: None,
            hidden_output: None,
            final_input: None,
            final_output: None,
        }
    }

    fn forward(&mut self, x: ArrayView<f64, Ix2>) -> Array2<f64> {
        let hidden_input = x.dot(&self.weight_input_hidden) + &self.bias_hidden;
        self.hidden_input = Some(hidden_input.clone());

        let hidden_output = relu(hidden_input);
        self.hidden_output = Some(hidden_output.clone());

        let final_input = hidden_output.dot(&self.weight_output_hidden) + &self.bias_output;
        self.final_input = Some(final_input.clone());

        let final_output = softmax(final_input);
        self.final_output = Some(final_output.clone());

        final_output
    }

    fn backward(&mut self, x: ArrayView<f64, Ix2>, y: ArrayView<f64, Ix2>, output: Array2<f64>) {
        let output_error = output - &y;
        let hidden_error = output_error.dot(&self.weight_output_hidden.t());
        let hidden_delta = hidden_error * relu_derivative(self.hidden_output.as_ref().unwrap());

        self.weight_output_hidden = &self.weight_output_hidden - self.learning_rate * self.hidden_output.as_ref().unwrap().t().dot(&output_error);
        self.bias_output = &self.bias_output - self.learning_rate * output_error.sum_axis(Axis(0)).insert_axis(Axis(0));

        self.weight_input_hidden = &self.weight_input_hidden - self.learning_rate * x.t().dot(&hidden_delta);
        self.bias_hidden = &self.bias_hidden - self.learning_rate * hidden_delta.sum_axis(Axis(0)).insert_axis(Axis(0));

        let l2_reg = 0.01;
        self.weight_output_hidden = &self.weight_output_hidden - l2_reg * &self.weight_output_hidden;
        self.weight_input_hidden = &self.weight_input_hidden - l2_reg * &self.weight_input_hidden;
    }

    fn train(&mut self, x: Array2<f64>, y: Array2<f64>, epochs: usize, batch_size: usize) {
        let num_samples = x.shape()[0];
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for i in (0..num_samples).step_by(batch_size) {
                let end = usize::min(i + batch_size, num_samples);
                let batch_x = x.slice(s![i..end, ..]);
                let batch_y = y.slice(s![i..end, ..]);

                let output = self.forward(batch_x.to_owned().view());
                self.backward(batch_x.to_owned().view(), batch_y.to_owned().view(), output.clone());

                let output_ln = output.mapv(|x| (x + 1e-8).ln());
                let product = batch_y.to_owned() * output_ln;
                total_loss -= product.sum();
            }
            println!("Epoch {}, Loss: {}", epoch, total_loss / num_samples as f64);
        }
    }

    fn predict(&mut self, x: Array2<f64>) -> Vec<usize> {
        let x_view: ArrayView2<f64> = x.view();
        let output = self.forward(x_view);
        output.axis_iter(Axis(0))
            .map(|row| row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0)
            .collect()
    }
}

fn main() {
    // Define the network structure and parameters
    let input_size = 2; // Number of input features
    let hidden_size = 5; // Number of hidden layer neurons
    let output_size = 2; // Number of output classes (for XOR, we need 2 classes: 0 and 1)
    let learning_rate = 0.1; // Learning rate

    // Create an MLP instance
    let mut mlp = MLP::new_mlp(input_size, hidden_size, output_size, learning_rate);

    // Define the XOR input data
    let input_data: Array2<f64> = array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ];

    // Define the XOR labels (one-hot encoded)
    let labels: Array2<f64> = array![
        [1.0, 0.0], // 0 XOR 0 = 0
        [0.0, 1.0], // 0 XOR 1 = 1
        [0.0, 1.0], // 1 XOR 0 = 1
        [1.0, 0.0]  // 1 XOR 1 = 0
    ];

    // Print the initial weights
    println!("Initial weights (input to hidden):\n{:?}", mlp.weight_input_hidden);
    println!("Initial weights (hidden to output):\n{:?}", mlp.weight_output_hidden);

    // Train the MLP with the XOR data
    let epochs = 10000; // Number of epochs
    let batch_size = 4; // Batch size (use the entire dataset as one batch)
    mlp.train(input_data.clone(), labels.clone(), epochs, batch_size);

    // Test the prediction
    let predictions = mlp.predict(input_data.clone());
    println!("Predictions: {:?}", predictions);

    // Print the final weights after training
    println!("Final weights (input to hidden):\n{:?}", mlp.weight_input_hidden);
    println!("Final weights (hidden to output):\n{:?}", mlp.weight_output_hidden);

    // Print the expected vs actual results for clarity
    println!("Expected: {:?}", labels);
    println!("Actual: {:?}", predictions);
}

