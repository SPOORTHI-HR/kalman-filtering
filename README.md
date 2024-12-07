# kalman-filtering
This repository contains a Python implementation of the Kalman filter, a powerful algorithm used for estimating the state of a dynamic system from a series of noisy measurements. The Kalman filter is widely used in various applications, including robotics, navigation, and finance.

### Kalman Filter Function

The `kalman_filter` function applies the Kalman filter to a series of measurements. It takes the following arguments:

- `measurements`: A NumPy array of shape (N, M) representing N measurements of M variables.
- `initial_state`: A NumPy array of shape (M,) representing the initial state estimate.
- `initial_covariance`: A NumPy array of shape (M, M) representing the initial state covariance.
- `process_noise`: A NumPy array of shape (M, M) representing the process noise covariance.
- `measurement_noise`: A NumPy array of shape (M, M) representing the measurement noise covariance.

The function returns a tuple containing:
- `filtered_states`: A NumPy array of shape (N, M) representing the filtered state estimates.
- `filtered_covariances`: A NumPy array of shape (N, M, M) representing the filtered state covariances.

### Implementation

```python
import numpy as np

def kalman_filter(measurements, initial_state, initial_covariance, process_noise, measurement_noise):
    """
    Applies the Kalman filter to a series of measurements.

    Args:
        measurements: A NumPy array of shape (N, M) representing N measurements of M variables.
        initial_state: A NumPy array of shape (M,) representing the initial state estimate.
        initial_covariance: A NumPy array of shape (M, M) representing the initial state covariance.
        process_noise: A NumPy array of shape (M, M) representing the process noise covariance.
        measurement_noise: A NumPy array of shape (M, M) representing the measurement noise covariance.

    Returns:
         A tuple containing:
             - filtered_states: A NumPy array of shape (N, M) representing the filtered state estimates.
             - filtered_covariances: A NumPy array of shape (N, M, M) representing the filtered state covariances.
    """

    # Initialize variables
    num_measurements = measurements.shape[0]
    num_variables = measurements.shape[1]
    filtered_states = np.zeros((num_measurements, num_variables))
    filtered_covariances = np.zeros((num_measurements, num_variables, num_variables))
    state = initial_state
    covariance = initial_covariance

    # Apply Kalman filter to each measurement
    for i in range(num_measurements):
        # Prediction step
        predicted_state = state # Assuming no state transition model, state remains same
        predicted_covariance = covariance + process_noise

        # Update step
        innovation = measurements[i] - predicted_state
        innovation_covariance = predicted_covariance + measurement_noise
        kalman_gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
        state = predicted_state + kalman_gain @ innovation
        covariance = (np.identity(num_variables) - kalman_gain) @ predicted_covariance

        # Store the filtered state and covariance
        filtered_states[i] = state
        filtered_covariances[i] = covariance

    return filtered_states, filtered_covariances
```

### Example Usage

Here's an example of how to use the `kalman_filter` function:

```python
import numpy as np

# Example measurements
measurements = np.array([[1.2], [1.5], [1.8]])

# Initial state estimate
initial_state = np.array([1.0])

# Initial state covariance
initial_covariance = np.array([[0.1]])

# Process noise covariance
process_noise = np.array([[0.01]])

# Measurement noise covariance
measurement_noise = np.array([[0.05]])

# Apply the Kalman filter
filtered_states, filtered_covariances = kalman_filter(measurements, initial_state, initial_covariance, process_noise, measurement_noise)

# Print the results
print("Filtered states:")
print(filtered_states)

print("\nFiltered covariances:")
print(filtered_covariances)
```

### Explanation

1. **Initialization**:
   - The function initializes the number of measurements and variables, as well as arrays to store the filtered states and covariances.
   - The initial state and covariance are set to the provided values.

2. **Prediction Step**:
   - The predicted state is assumed to remain the same (no state transition model).
   - The predicted covariance is updated by adding the process noise covariance.

3. **Update Step**:
   - The innovation (difference between the measurement and predicted state) is calculated.
   - The innovation covariance is updated by adding the measurement noise covariance.
   - The Kalman gain is calculated using the predicted covariance and innovation covariance.
   - The state is updated using the Kalman gain and innovation.
   - The covariance is updated using the Kalman gain and predicted covariance.

4. **Storing Results**:
   - The filtered state and covariance are stored for each measurement.

This implementation provides a simple yet effective way to apply the Kalman filter to a series of measurements. Feel free to customize and extend the code to suit your specific needs.
