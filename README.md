# Gradient Descent Experiments

This repository contains a script to run multiple instances of Gradient Descent and its variants on the MNIST dataset 
using different configurations and compare between them. 

The script allows you to specify various parameters for each run, making it flexible for different experiments.

## Prerequisites

Install the project requirements with either requirements.txt (pip) or environment.yml (conda).
## How to Run the Script

To run the `main.py` script, you will use command-line arguments. Here’s a breakdown of how to format your command and the available options.

### Command Format

```bash
python main.py -s SEED -v VARIANT --digits DIGITS --loss LOSS_FUNCTION --epochs EPOCHS --rate LEARNING_RATE [OPTIONS]
```

### Required Arguments

- **`-s` or `--seed`**: 
  - Type: Integer
  - Description: Seed for the random number generator. Can be specified multiple times.
  - Example: `-s 42`

- **`-v` or `--variant`**: 
  - Type: String
  - Description: The names of the Gradient Descent variants to run. Can be specified multiple times. 
  - One of {'GD': (vanilla) gradient descent, 'SGD': stochastic, 'RGD': regularized, 'PGD':projected, 'CGD':constrained}
  - Example: `-v SGD`

- **`--digits`**: 
  - Type: Integer pairs
  - Description: The classes for binary classification (0-9). Can be specified multiple times.
  - Example: `--digits 0 9`

- **`--loss` or `--loss_function`**: 
  - Type: String
  - Description: The loss function to use for optimization. Can be specified multiple times.
  - One of {'hinge', 'BCE'}
  - Example: `--loss hinge`

- **`--epochs`**: 
  - Type: Integer
  - Description: Number of epochs to run the Gradient Descent variant. Can be specified multiple times.
  - Example: `--epochs 100`

- **`--rate` or `--learning_rate`**: 
  - Type: Float
  - Description: The learning rate to use for updating weights. Can be specified multiple times.
  - Example: `--rate 0.01`

### Optional Arguments

- **`--coefficient`**: 
  - Type: Float
  - Description: The regularization coefficient to be used in the variant RGD. Can be specified multiple times.
  
- **`--radius`**: 
  - Type: Float
  - Description: The radius of the hypersphere that weights will be projected into in the variant PGD. Can be specified multiple times.
  
- **`--cutoff`**: 
  - Type: Float
  - Description: The cutoff value to clip the weight vector in the variant CGD. Can be specified multiple times.

- **`--compare`**: 
  - Type: Flag
  - Description: Generate additional graphs for comparing the loss and accuracy against the number of iterations. May only be specified once.
  
- **`--verbose`**: 
  - Type: Flag
  - Description: Prints additional information and details. May only be specified once.

### Example Command

Here's an example command that demonstrates how to use the script:

```bash
python main.py -s 42 -v GD -d 1 5 --loss BCE --epochs 100 --rate 0.1  -s 42 -v RGD -d 1 5 --loss BCE --epochs 100 --rate 0.1 --coefficient 0.01 --compare
```

### Notes

- Ensure that the number of `--epochs`, `--rate`, and other parameters match the number of variants specified.
- You can specify multiple values for parameters by repeating the argument.

## Conclusion

You can customize your Gradient Descent experiments by changing the parameters in the command line. Make sure to refer to the descriptions for each argument to set them appropriately. Happy experimenting!