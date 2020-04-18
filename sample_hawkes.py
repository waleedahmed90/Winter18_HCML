#!/usr/bin/env python
import click
import numpy as np

#Sources used for Help
#1#https://radhakrishna.typepad.com/mle-of-hawkes-self-exciting-process-1.pdf
#2#https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf

@click.command()
@click.argument('mu', type=float)
@click.argument('a', type=float)
@click.argument('w', type=float)
@click.argument('T', type=float)
@click.option('--max-events', 'max_events', default=1000000, help='Maximum number of events to generate before sampling termination.')
@click.option('--N', 'N', default=1, help='Number of time sequences to generate.', show_default=True)
@click.option('--seed', 'seed', help='Initial seed. A random sequence is generated if seed is -1.', default=-1, show_default=True)


def run(mu, a, w, t, N, seed, max_events):
    """Generates a sample from Hawkes process with the given MU, A, and W parameters.

    A sequence upto time T is generated, unless max-events are reached, in which case the
    program stops with an error."""

    T = t

    if seed < 0:
        seed = np.random.randint(10000) + 1

    for i in range(N):
        print(','.join([str(x)
                        for x in sample_hawkes(mu=mu, a=a, w=w, T=T,
                                               seed=seed + i,
                                               max_events=max_events)]))


def sample_hawkes(mu, a, w, T, seed, max_events):
    
    np.random.seed(seed)
    tev = np.zeros(max_events)-1

    t = t_i = i = 0

    lambda_max = mu+(a*np.exp(-w*(t-t_i)))

    u1 = np.random.uniform(0,1)
    t = t - np.log(1-u1)/lambda_max
    t_i = t
    #print ("lambda_max: ", lambda_max )

    dlambda_history = a

    tev[i] = t_i
    i +=1

    while t<T:
        t_prev = t_i
        u1 = np.random.uniform(0,1)
        t = t - np.log(1-u1)/lambda_max
        u2 = np.random.uniform(0,1)

        if (u2 <= (mu + dlambda_history*np.exp(-w*(t-t_i)))/lambda_max):# or (t<T):
            dlambda_history = a + dlambda_history*np.exp(-w*(t-t_i))
            
            lambda_max = lambda_max + a*np.exp(-w*(t_i-t_prev))
            
            ##lambda_max += a
            #print ("lambda_max: ", lambda_max )
            t_i = t
            tev[i] = t_i
            i += 1

    tev = tev[1:i-1]
    return tev
            

if __name__ == '__main__':
    run()
