import time

class PID:
    """PID Controller"""

    def __init__(self, P = 0.0, I = 0.0, D = 0.0, min_v = 0.0, max_v = 0.0):

        self.k_p = P
        self.k_i = I
        self.k_d = D

        self.min_ = min_v
        self.max_ = max_v

        self.current_time = time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.ref = 0.0
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0
        self.last_error = 0.0
        self.command = 0.0

    def getCmd(self, meas):
        """Calculates PID value for given reference feedback"""
        error = self.ref - meas
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        """prevent zero division"""
        if(delta_time <= 0.0):
            raise ValueError('delta_time less than or equal to 0')

        """Integrate"""
        self.i_error += delta_time*k_i*error

        """Anti-windup"""
        self.i_error += min(max(i_error, min_),max_)

        """p and d error"""
        self.p_error = error
        self.d_error = delta_error / delta_time

        """calculate command"""
        self.command = (self.k_p*self.p_error) + (self.k_i * self.i_error) + (self.k_d * self.d_error)
        self.command = min(max(self.output, min_ ), max_)

        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = error



    def setKp(self, proportional_gain):
        """Proportional Gain"""
        self.k_p = proportional_gain

    def setKi(self, integral_gain):
        """Integral Gain"""
        self.k_i = integral_gain

    def setKd(self, derivative_gain):
        """Derivative Gain"""
        self.k_d = derivative_gain

    def setMinMax(self,min_value, max_value):
        """set min and max clampping values """
        self.min_ = min_value
        self.max_ = max_value

    def setRef(self, reference):
        """set reference value"""
        self.ref = reference
