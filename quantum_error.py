import classical
from scipy.stats import binom
import numpy as np

def sim_prediction(pred_y, true_y, n_shots):
    # Suppose we want to predict y, what's the MSE based on the device?
    optimal_p = np.arctan(pred_y) / 2.75 + .5
    assert(0 <= optimal_p <= 1)
    losses = np.zeros(n_shots)
    probs = np.zeros(n_shots)
    for num_ones in range(n_shots):
        prediction = np.tan((num_ones/n_shots - .5) * 2.75)
        probs[num_ones] = binom.pmf(num_ones, n_shots, optimal_p)
        losses[num_ones] = (prediction - true_y) ** 2
    avg_error = np.sum(losses * probs)
    return avg_error
    
if __name__ == "__main__":
    sequence_length = 5
    x_train, y_train, x_val, y_val, x_test, y_test = \
        classical.get_data(sequence_length, n_stocks_train=100, n_stocks_val=64)
    opt_pred_value = classical.contant_baseline(y_train, y_val, y_test)
    for n_shots in [100, 300, 1000]:
        for y in range(0, 6, 1):
            err = sim_prediction(y, y, n_shots)
            print(f"Want to predict {y} with {n_shots}, error is {err}")
        for name, test_set in ("Train", y_train), ("Val", y_val), ("Test", y_test):
            losses = []
            for test_point in test_set:
                losses.append(sim_prediction(opt_pred_value, test_point[-1], n_shots))
            print(f"Avg {name} error {np.mean(losses)}")
