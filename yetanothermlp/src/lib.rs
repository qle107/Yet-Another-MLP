extern crate core;

use ndarray::{Array2, ArrayView2, Axis, s};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::thread_rng;
use std::ffi::c_void;
use std::os::raw::{c_double, c_int};

#[repr(C)]
pub struct MLP {
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
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> MLP {
        let rng = thread_rng();
        let weight_input_hidden: Array2<f64> = Array2::random((input_size, hidden_size), StandardNormal) * 0.01;
        let weight_output_hidden: Array2<f64> = Array2::random((hidden_size, output_size), StandardNormal) * 0.01;
        let bias_hidden: Array2<f64> = Array2::zeros((1, hidden_size));
        let bias_output: Array2<f64> = Array2::zeros((1, output_size));

        MLP {
            weight_input_hidden,
            weight_output_hidden,
            bias_hidden,
            bias_output,
            learning_rate,
            hidden_input: None,
            hidden_output: None,
            final_input: None,
            final_output: None,
        }
    }

    fn forward(&mut self, x: ArrayView2<f64>) -> Array2<f64> {
        let hidden_input = x.dot(&self.weight_input_hidden) + &self.bias_hidden;
        self.hidden_input = Some(hidden_input.clone());

        let hidden_output = hidden_input.mapv(|a| a.max(0.0));
        self.hidden_output = Some(hidden_output.clone());

        let final_input = hidden_output.dot(&self.weight_output_hidden) + &self.bias_output;
        self.final_input = Some(final_input.clone());

        let max = final_input.map_axis(Axis(1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
            .insert_axis(Axis(1));
        let exp_x = (&final_input - &max).mapv(f64::exp);
        let sum_exp_x = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));
        let final_output = exp_x / &sum_exp_x;
        self.final_output = Some(final_output.clone());

        final_output
    }

    unsafe fn backward(&mut self, x: ArrayView2<f64>, y: ArrayView2<f64>, output: Array2<f64>) {
        let hidden_output = self.hidden_output.as_ref().unwrap();
        let weight_output_hidden = &self.weight_output_hidden;

        let mut output_error = output - &y;
        let mut hidden_error = output_error.dot(&weight_output_hidden.t());
        let relu_derivative = hidden_output.mapv(|a| if a > 0.0 { 1.0 } else { 0.0 });
        hidden_error = hidden_error * relu_derivative;

        let weight_output_update = hidden_output.t().dot(&output_error) * self.learning_rate;
        let weight_input_update = x.t().dot(&hidden_error) * self.learning_rate;

        let mut bias_output_update = output_error.sum_axis(Axis(0));
        let mut bias_hidden_update = hidden_error.sum_axis(Axis(0));

        bias_output_update *= self.learning_rate;
        bias_hidden_update *= self.learning_rate;

        self.weight_output_hidden -= &weight_output_update;
        self.weight_input_hidden -= &weight_input_update;
        self.bias_output -= &bias_output_update.insert_axis(Axis(0));
        self.bias_hidden -= &bias_hidden_update.insert_axis(Axis(0));
    }

    unsafe fn train(&mut self, x: ArrayView2<f64>, y: ArrayView2<f64>, epochs: usize, batch_size: usize) {
        let num_samples = x.shape()[0];
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;

            for i in (0..num_samples).step_by(batch_size) {
                let end = usize::min(i + batch_size, num_samples);
                let batch_x = x.slice(s![i..end, ..]);
                let batch_y = y.slice(s![i..end, ..]);

                let output = self.forward(batch_x.to_owned().view());
                self.backward(batch_x.to_owned().view(), batch_y.to_owned().view(), output.clone());

                let output_ln = output.mapv(|x| (x + 1e-8).ln());
                let product = batch_y.to_owned() * output_ln;
                total_loss -= product.sum();
                let mut predictions = Vec::new();
                for row in output.axis_iter(Axis(0)) {
                    let mut max_value = row[0];
                    let mut max_index = 0;
                    for (i, &value) in row.iter().enumerate() {
                        if value > max_value {
                            max_value = value;
                            max_index = i;
                        }
                    }
                    predictions.push(max_index);
                }

                let mut true_labels = Vec::new();
                for row in batch_y.axis_iter(Axis(0)) {
                    let mut max_value = row[0];
                    let mut max_index = 0;
                    for (i, &value) in row.iter().enumerate() {
                        if value > max_value {
                            max_value = value;
                            max_index = i;
                        }
                    }
                    true_labels.push(max_index);
                }

                let mut correct_predictions = 0;
                for (pred, true_label) in predictions.iter().zip(true_labels.iter()) {
                    if pred == true_label {
                        correct_predictions += 1;
                    }
                }
            }
            let accuracy = correct_predictions as f64 / num_samples as f64 * 100.0;

            println!("Epoch {}, Loss: {}, Accuracy: {:.3}%", epoch, total_loss / num_samples as f64, accuracy);
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

#[no_mangle]
pub extern "C" fn mlp_new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: c_double) -> *mut MLP {
    let mlp = MLP::new(input_size, hidden_size, output_size, learning_rate);
    Box::into_raw(Box::new(mlp))
}

#[no_mangle]
pub extern "C" fn mlp_forward(mlp: *mut MLP, x: *const c_double, rows: usize, cols: usize) -> *mut c_double {
    let mlp = unsafe { &mut *mlp };
    let x = unsafe { ArrayView2::from_shape_ptr((rows, cols), x) };
    let output = mlp.forward(x);
    let output_ptr = output.as_ptr();
    std::mem::forget(output); // Prevents Rust from freeing the array
    output_ptr as *mut c_double
}

#[no_mangle]
pub extern "C" fn mlp_train(mlp: *mut MLP, x: *const c_double, y: *const c_double, x_rows: usize, x_cols: usize, y_rows: usize, y_cols: usize, epochs: usize, batch_size: usize) {
    let mlp = unsafe { &mut *mlp };
    let x = unsafe { ArrayView2::from_shape_ptr((x_rows, x_cols), x) };
    let y = unsafe { ArrayView2::from_shape_ptr((y_rows, y_cols), y) };
    unsafe {
        mlp.train(x, y, epochs, batch_size);
    }
}

#[no_mangle]
pub extern "C" fn mlp_free(mlp: *mut MLP) {
    if !mlp.is_null() {
        unsafe {
            Box::from_raw(mlp);
        }
    }
}
