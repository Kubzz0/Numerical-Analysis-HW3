####################
#### Question 1 ####
####################
import numpy as np
import matplotlib.pyplot as plt

class LinearizedModelFitter:
    def __init__(self, x, y):
        ################################################
        #### Initialise the Class with x and y data ####
        ################################################
        self.x = np.array(x)
        self.y = np.array(y)
        self.params = None
        self.predicted_y = None

    def linearize(self):
        ##################################################
        #### Linearize: ln(y) = ln(α) + ln(x) + β * x ####
        ##################################################
        
        # Transform the data: ln(y) = ln(α) + ln(x) + β * x
        ln_y = np.log(self.y)  # Log-transform y
        ln_x = np.log(self.x)  # Log-transform x
        
        # Perform linear regression: ln(y) = ln(α) + β * x + ln(x)
        x = self.x
        y = ln_y - ln_x  # Transform the dependent variable
        
        # Compute means
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate slope (β) and intercept (ln(α)) using least squares formulas
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean

        self.params = {"alpha": np.exp(intercept), "beta": slope}

        # Calculate predicted y values using the original model y = α * x * e^(β * x)
        alpha, beta = self.params["alpha"], self.params["beta"]
        self.predicted_y = alpha * self.x * np.exp(beta * self.x)

    def plot_results(self):
        ##############################################################
        #### Plots the Fitted Curve Along with the original data ####
            #### Plots the residuals to assess the model's fit ####
        #############################################################
        if self.params is None:
            raise ValueError("Model parameters not fitted yet.")
        
        # Fitted y vs. x plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(self.x, self.y, label="Data", color="blue")
        x_fit = np.linspace(min(self.x), max(self.x), 100)
        y_fit = self.predict(x_fit)
        plt.plot(x_fit, y_fit, label="Fitted Model", color="red")
        plt.title("Fitted y vs x")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

        # Residuals vs. Predicted y plot
        plt.subplot(1, 2, 2)
        residuals = self.y - self.predicted_y
        plt.scatter(self.predicted_y, residuals, color="purple")
        plt.axhline(0, color="black", linestyle="--")
        plt.title("Residuals vs Predicted y")
        plt.xlabel("Predicted y")
        plt.ylabel("Residuals")
        plt.tight_layout()
        plt.show()

    def predict(self, x):
        ########################################################
        #### Predict using the model y = a * x * exp(b * x) ####
        ########################################################
        if self.params is None:
            raise ValueError("Model parameters not fitted yet.")
        alpha, beta = self.params["alpha"], self.params["beta"]
        return alpha * x * np.exp(beta * x)

# Data for Question 1
x1 = [0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.5, 1.7, 1.8]
y1 = [0.75, 1.25, 1.45, 1.25, 0.85, 0.55, 0.35, 0.28, 0.18]

# Solve Question 1
model = LinearizedModelFitter(x1, y1)
model.linearize()
model.plot_results()
print("Question 1 Parameters:", model.params)
