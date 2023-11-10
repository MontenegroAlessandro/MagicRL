"""
Summary: ADAM implementation
Author: @MontenegroAlessandro
Date: 31/10/2023
"""

# Libraries
import numpy as np


# Adam class implementation
class Adam:
    """Class implementing Adam learning rate optimizer"""
    def __init__(self, step_size: float = 1e-3, strategy: str = "descent"):
        """
        Summary:
            initialization
        Args:
            step_size (float): the adam learning rate.
            Default to 1e-3.

            strategy (str): the update strategy, it can be "descent" or "ascent".
            Default to "descent".
        """
        # classical Adam parameter
        # fixme -> make betas settable
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.m = 0
        self.v = 0
        self.t = 1
        self.epsilon = 1e-8

        # get the step size
        err_msg = "[ADAM] the step size must be positive!"
        assert step_size >= 0, err_msg
        self.step_size = step_size

        # get the strategy
        err_msg = "[ADAM] strategy must be \'ascent\' or \'descent\'!"
        assert strategy in ["ascent", "descent"], err_msg
        self.strategy = strategy

        # revert the step size sign
        if self.strategy == "ascent":
            # self.step_size = -self.step_size
            self.step_size = self.step_size

    def compute_gradient(self, g: np.array) -> np.array:
        """
        Summary:
            this function computes the new gradient with the adaptive learning
            rate.
        Args:
            g (np.array): the gradient already computed by the algorithm.
        Returns:
            np.array: the modified gradient.
        """
        # compute m and v
        self.m = self.m * self.beta_1 + (1 - self.beta_1) * g
        self.v = self.v * self.beta_2 + (1 - self.beta_2) * np.power(g, 2)

        # compute m_hat and v_hat
        m_hat = self.m / (1 - np.power(self.beta_1, self.t))
        v_hat = self.v / (1 - np.power(self.beta_2, self.t))

        # compute the new gradient
        new_gradient = self.step_size * m_hat / (self.epsilon + np.sqrt(v_hat))

        # update t
        self.t += 1

        return new_gradient
