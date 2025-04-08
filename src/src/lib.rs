extern crate ndarray;
extern crate rand;

use ndarray::{Array, Array1, Array2, ArrayView1, Axis};
use rand::distributions::Uniform;
use rand::prelude::*;

type Float = f32;

struct MLP {
    weights_input_hidden: Array2<Float>,
    weights_hidden_output: Array2<Float>,
    bias_hidden: Array1<Float>,
    bias_output: Array1<Float>,
    learning_rate: Float,
}

impl MLP {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: Float) -> Self {
        let mut rng = thread_rng();
        let input_hidden_range = Uniform::new(-0.01, 0.01);
        let hidden_output_range = Uniform::new(-0.01, 0.01);

        MLP {
            weights_input_hidden: Array::random_using((input_size, hidden_size), input_hidden_range, &mut rng),
            weights_hidden_output: Array::random_using((hidden_size, output_size), hidden_output_range, &mut rng),
            bias_hidden: Array::zeros(hidden_size),
            bias_output: Array::zeros(output_size),
            learning_rate,
        }
    }

    fn relu(x: &Array2<Float>) -> Array2<Float> {
        x.mapv(|a| if a > 0.0 { a } else { 0.0 })
    }

    fn relu_derivative(x: &Array2<Float>) -> Array2<Float> {
        x.mapv(|a| if a > 0.0 { 1.0 } else { 0.0 })
    }

    fn softmax(x: &Array2<Float>) -> Array2<Float> {
        x.map_axis(Axis(1), |row| {
            let exp_row = row.mapv(|a| a.exp());
            &exp_row / exp_row.sum()
        })
    }

    fn forward(&mut self, x: &Array2<Float>) -> Array2<Float> {
        let hidden_input = x.dot(&self.weights_input_hidden) + &self.bias_hidden;
        let hidden_output = MLP::relu(&hidden_input);
        let final_input = hidden_output.dot(&self.weights_hidden_output) + &self.bias_output;
        MLP::softmax(&final_input)
    }

    fn backward(&mut self, x: &Array2<Float>, y: &Array2<Float>, output: &Array2<Float>) {
        let output_error = output - y;
        let hidden_output = MLP::relu(&(x.dot(&self.weights_input_hidden) + &self.bias_hidden));
        let hidden_error = output_error.dot(&self.weights_hidden_output.t());
        let hidden_delta = hidden_error * MLP::relu_derivative(&hidden_output);

        self.weights_hidden_output -= &hidden_output.t().dot(&output_error) * self.learning_rate;
        self.bias_output -= output_error.sum_axis(Axis(0)) * self.learning_rate;

        self.weights_input_hidden -= &x.t().dot(&hidden_delta) * self.learning_rate;
        self.bias_hidden -= hidden_delta.sum_axis(Axis(0)) * self.learning_rate;

        let l2_reg = 0.01;
        self.weights_hidden_output -= &self.weights_hidden_output * l2_reg;
        self.weights_input_hidden -= &self.weights_input_hidden * l2_reg;
    }

    fn train(&mut self, x: &Array2<Float>, y: &Array2<Float>, epochs: usize, batch_size: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for i in (0..x.len_of(Axis(0))).step_by(batch_size) {
                let batch_x = x.slice(s![i..i + batch_size, ..]);
                let batch_y = y.slice(s![i..i + batch_size, ..]);
                let output = self.forward(&batch_x.to_owned());
                self.backward(&batch_x.to_owned(), &batch_y.to_owned(), &output);
                total_loss += -(&batch_y * &output.mapv(|o| o.ln())).sum();
            }
            println!("Epoch {}, Loss: {}", epoch, total_loss / x.len_of(Axis(0)) as Float);
        }
    }

    fn predict(&mut self, x: &Array2<Float>) -> Array1<usize> {
        let output = self.forward(x);
        output.map_axis(Axis(1), |row| row.argmax().unwrap())
    }
}

fn one_hot_encode(labels: &Array1<usize>, num_classes: usize) -> Array2<Float> {
    let mut one_hot = Array2::<Float>::zeros((labels.len(), num_classes));
    for (i, &label) in labels.iter().enumerate() {
        one_hot[(i, label)] = 1.0;
    }
    one_hot
}
