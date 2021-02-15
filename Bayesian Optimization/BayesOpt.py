import numpy as np
import matplotlib.pyplot as plt


from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from scipy.stats import norm



class BayesOptViz:

    def __init__(self, f_obj, f, upper, lower, N_max, n_init, kernel):
        """Simple class in order to represent the behavior of a
        bayesian optimization process.

        Parameters
        ----------
        f_obj : function
            objective function
        f : function
            noisy objective function.
        upper : float
            maximum value of x.
        lower : type
            minimum value of x.
        N_max : type
            maximal number of optimization step.
        n_init : type
            Number of initials points to evaluate.
        kernel : type
            Kernel used for the gaussian process model

        Returns
        -------
        type
            Description of returned object.

        """

        self.f_obj = f_obj # objective function
        self.f = f # noisy objective function
        self.lower, self.upper = lower, upper # range of x
        self.N_max = N_max # maximal iteration number
        self.kernel = kernel # kernel used for the gaussian process
        self.x = list(np.linspace(lower, upper, n_init)) # initials points for evaluation
        self.y = [self.f_obj(z) for z in self.x] # evaluations of the objective function on the initials points
        self.model = GPR(kernel=self.kernel, normalize_y=True).fit(np.array(self.x).reshape(-1, 1), np.array(self.y).reshape(-1, 1)) # Gaussian process model

    def f_approx(self, x, return_std=False):
        """Computation of the acquisition function on a point x.

        Parameters
        ----------
        x : float
            point of the evaluation, x should belong to [self.lower, self.upper].
        return_std : Boolean
            return std if True.

        Returns
        -------
        tuple
            (value, std) if return_std else value. where value is the gaussian process prediction

        """
        return self.model.predict(x, return_std=return_std)

    def probs(self, sample):
        """Compute probability of improvement for a given sample of points x

        Parameters
        ----------
        sample : list
            list of points where to compute the expected probability.

        Returns
        -------
        numpy.array
            array containing probabilities of improvement.

        """
        best = min(self.f_approx(np.array(self.x).reshape(-1, 1)))
        y_sample, std = self.f_approx(np.array(sample).reshape(-1, 1), return_std=True)
        y_sample = y_sample[:, 0]
        probs = norm.cdf((best-y_sample) / (std+1E-9))

        return probs

    def next_point(self):
        """Choose next point for evaluation based on probability of improvements.
        Add this point to the x attribut and add the evaluation of the objective
        function to the y attribut.


        Returns
        -------
        None

        """
        sample = [np.random.rand() * (self.upper-self.lower) + self.lower for i in range(100)]
        probs = self.probs(sample)
        idx = np.argmax(probs)
        self.x.append(sample[idx])
        self.y.append(self.f_obj(sample[idx]))

    def plot(self):
        """plot the acquisition function with the confidence interval and the objective function.

        Returns
        -------
        None

        """
        fig, axs = plt.subplots(2, figsize=(20,10))
        x_plot = np.linspace(self.lower, self.upper, 300)
        y_real = [self.f(z) for z in x_plot]
        y_approx, sigma = self.f_approx(np.array(x_plot).reshape(-1, 1), return_std=True)
        axs[0].plot(x_plot, y_real, 'r:', label='f(x)')
        axs[0].plot(self.x, self.y, 'r.', markersize=10, label='Observations')
        axs[0].plot(x_plot, y_approx, 'b-', label='Prediction')
        axs[0].fill_between(x_plot, (y_approx.reshape(len(y_approx),) - 1.9600 * sigma),
                         (y_approx.reshape(len(y_approx),) + 1.9600 * sigma), color='b', alpha=.4,
                         label='95% confidence interval')
        axs[0].set(xlabel='x', ylabel='f(x)')
        axs[0].legend(loc='upper left')
        y_probs = self.probs(x_plot)
        axs[1].plot(x_plot, y_probs)

    def n_step(self,n=1 , plot=False):
        """Compute one optimization step.

        Parameters
        ----------
        plot : Boolean
            if True, plot the acquisition function with confidence interval and
            the objective function when the optimization step is done.

        Returns
        -------
        None

        """
        for i in range(n):
            if len(self.x) < self.N_max:
                self.next_point()
                self.model = GPR(kernel=self.kernel, normalize_y=True).fit(np.array(self.x).reshape(-1, 1), np.array(self.y).reshape(-1, 1))
            else:
                print("Optimisation terminee")
        if plot:
            self.plot()

    def get_min(self):
        """Return the minimun of self.y and the corresponding x .

        Returns
        -------
        tuple (float, float)
                (argmin(f(x)) , min(f(x)))

        """

        """Recherche du minimum global calculé ainsi que ses coordonnées"""
        idx = np.argmin(self.y)
        return self.x[idx], self.y[idx]

    def complete_run(self):
        """Run optimization step until N_max points have been evaluated.

        Returns
        -------
        tuple (float, float)
                (argmin(f(x)) , min(f(x)))

        """
        while len(self.x) < self.N_max:
            self.n_step()
        return self.get_min()
