#!/usr/bin/env python
import click
import numpy as np
import cvxpy as CVX


def Hawkes_log_lik(T, w, alpha_var, mu_var, tevs, for_cvx=True):
    """Calculate the likelihood of the event sequences given in tev."""
    objective = CVX.Variable()
    objective = 0
    Integral_term = 0
    
    sum_i = 0
    for tev in tevs:
        #hist_j = 0
        LL = 0
        for i in range(len(tev)):
            hist_j = 0
            for j in range(i):
                hist_j += np.exp(-w * (tev[i] - tev[j]))
                
            Integral_term = (alpha_var/w)*(CVX.exp(w*tev[i]))*(CVX.exp(-w*T)-CVX.exp(-w*tev[i]))
            LL += CVX.log(mu_var + (alpha_var * hist_j)) + Integral_term 
            
        objective += LL - (mu_var*T)
    return objective    


@click.command()
@click.argument('w', type=float)
@click.argument('T', type=float)
@click.argument('seq_file', type=click.File('r'))
def run(w, t, seq_file):
    """Read samples of events generated by a Hawkes process from SEQ_FILE
    as one sample per line.

    Then, assuming that the Hawkes process has decay rate 'w' and the
    observations were all stopped at time 'T', output the inferred values of
    'a' and 'mu'."""

    T = t

    # Read sequences from file into a list of numpy arrays
    print('Reading data ...')
    tevs = [np.asarray([float(x) for x in line.split(',')])
            for line in seq_file]

    
    
    # Set up CVX problem
    alpha_var = CVX.Variable()
    constraints = [alpha_var >= 0]

    mu_var = CVX.Variable()
    constraints.append(mu_var >= 0)

    prob = CVX.Problem(
        CVX.Maximize(Hawkes_log_lik(T=T,
                                    w=w,
                                    alpha_var=alpha_var,
                                    mu_var=mu_var,
                                    tevs=tevs)),
        constraints=constraints
    )

    result = prob.solve(verbose=True)

    print('Inferred values:')
    print('\ta = {}'.format(alpha_var.value))
    print('\tmu = {}'.format(mu_var.value))
    print('\tLL = {}'.format(result))


if __name__ == '__main__':
    run()