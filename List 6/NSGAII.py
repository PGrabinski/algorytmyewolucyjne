from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
from pymoo.core.problem import Problem
import zipfile
import pandas as pd


def load_stock_quotations(stock_names, filename):
    s = {}
    with zipfile.ZipFile(filename) as z:
        for stock_name in stock_names:
            with z.open(stock_name + '.mst') as f:
                s[stock_name] = pd.read_csv(f, index_col='<DTYYYYMMDD>', parse_dates=True)[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
                s[stock_name].index.rename('time', inplace=True)
                s[stock_name].rename(columns={'<OPEN>':'open', '<HIGH>':'high', '<LOW>':'low', '<CLOSE>':'close', '<VOL>':'volume'}, inplace=True)
    return pd.concat(s.values(), keys=s.keys(), axis=1)

STOCK_QUOTATIONS_ARCHIVE_FILE_NAME = 'mstall.zip'
STOCK_NAMES_FILE_NAME = 'WIG20.txt'


class WIG20PortfolioOptimizationProblem(Problem):

    def __init__(self, delta_t=90):
        super().__init__(n_var=20, n_obj=2, n_ieq_constr=1, xl=0., xu=1.)
        self.stock_names = pd.read_csv(STOCK_NAMES_FILE_NAME, index_col=0, names=['stock name']).index.values
        self.stock_names.sort()
        self.number_of_stocks = len(self.stock_names)

        self.stock_quotations = load_stock_quotations(self.stock_names, STOCK_QUOTATIONS_ARCHIVE_FILE_NAME).dropna()

        self.delta_t = delta_t
        self.stock_returns = self.stock_quotations.xs('close', level=1, axis=1).pct_change()
        self.stock_returns_m = self.stock_returns[-self.delta_t-1:-1].mean()
        self.stock_returns_s = self.stock_returns[-self.delta_t-1:-1].std()
        self.stock_covariances = self.stock_returns[-self.delta_t-1:-1].cov()
        self.stock_correlations = self.stock_returns[-self.delta_t-1:-1].corr()

    def _evaluate(self, p, out, *args, **kwargs):
        normalization = 1 - p.sum(axis=1)
        out["G"] = normalization
        returns = -1 * p @ self.stock_returns_m.to_numpy()

        # (population_size, number_of_stocks) @ (number_of_stocks, number_of_stocks) = (population_size, timesteps)
        portfolio_value = p @ self.stock_returns.T.to_numpy()
        # (population_size, number_of_stocks) @ (number_of_stocks, ) = (population_size, )
        portfolio_mean_values = p @ self.stock_returns_m.to_numpy()
        
        semivariance = np.zeros((p.shape[0],))
        for i in range(p.shape[0]):
            portfolio_negative_values_i = portfolio_value[i, :]
            portfolio_negative_values_i = portfolio_negative_values_i[portfolio_negative_values_i < portfolio_mean_values[i]]
            semivariance[i] = ((portfolio_negative_values_i - portfolio_mean_values[i]) ** 2).mean()

        
        objective = np.column_stack([semivariance, returns])

        out["F"] = objective


# problem = WIG20PortfolioOptimizationProblem()

# algorithm = NSGA2(pop_size=1000)

# res = minimize(problem,
#                algorithm,
#                ('n_gen', 1000),
#                seed=42,
#                verbose=False)

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, facecolor="none", edgecolor="red")
# plot.save("pareto_front.png")