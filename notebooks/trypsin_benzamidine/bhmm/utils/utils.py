import numpy as np
import matplotlib.pyplot as plt
from pyemma.util.statistics import confidence_interval



def units():
    kB=1.9872066124357268*1e-3    # kcal/mol
    T=300.0                       # K
    beta=1.0/(kB*T)               # mol/kcal
    return kB, T, beta    


def save_figure(name):
    # change these if wanted
    do_save = True
    fig_dir = './figs/'
    if do_save:
        plt.savefig(fig_dir + name, bbox_inches='tight')


def index_unbound_bound(bhmm, i_micro_unbound):
    """
    The metastable distributions (mm.metastable_distributions), or equivalently the observation probabilities are the probability 
    to be in a given cluster (‘microstate’) if we are in one of the hidden metastable states.
    """
    i_bound = np.argmax(bhmm.stationary_distribution)
    i_unbound = np.argmax(bhmm.observation_probabilities[:, i_micro_unbound])
    
    if i_bound == i_unbound:
        print("Bound and unbound found in same macrostate. Bound state will be redefined.")
        #i_unbound = np.argsort(M.observation_probabilities[:, i_micro_unbound])[-2]
        i_bound = np.argsort(bhmm.stationary_distribution)[-2]
        
    return i_unbound, i_bound


def pi2dG(pi, i_unbound, i_bound):
    #kB, T, beta = units()
    delta_g = -0.6 * np.log(pi[i_bound]/pi[i_unbound])  # dG in kcal/mol
    delta_g -= 3.1  # volume correction to standard binding free energy
    return delta_g


def binding_free_energy(bhmm, i_micro_unbound, conf=0.95):
    i_unbound, i_bound = index_unbound_bound(bhmm, i_micro_unbound)    

    # MLE
    pi_mle = bhmm.stationary_distribution
    dG_mle = pi2dG(pi_mle, i_unbound, i_bound)
    print(">lag={}: dG_mle={:.2f} / pi_mle={} / (i_unbound: {:d}, i_bound: {:d})".format(bhmm.lag, dG_mle, pi_mle, i_unbound, i_bound))
    
    # samples (gibbs sampling)
    try:
        pi_samples = bhmm.sample_f('stationary_distribution')
        dG_samples = [pi2dG(pi_sample, i_unbound, i_bound) for pi_sample in pi_samples]
        l, r = confidence_interval(dG_samples, conf=conf)
    except:
        l, r = 0, 0
        
    return dG_mle, l, r


def mfpt2kon(mfpt):
    mfpt *= 1e-9  # in seconds
    # volume fraction
    Nsim = 10604.0  # number of water molecules in our simulation
    Nstd = 55.55  # number of water molecules in standard volume
    concentration = Nstd / Nsim
    return 1./(mfpt*concentration)


def binding_rate(bhmm, i_micro_unbound, conf=0.95):
    i_unbound, i_bound = index_unbound_bound(bhmm, i_micro_unbound)
    # MLE
    mfpt_mle = bhmm.mfpt(i_unbound, i_bound)
    kon = mfpt2kon(mfpt_mle)
    # samples
    mfpt_samples = bhmm.sample_f('mfpt', i_unbound, i_bound)
    kon_samples = [mfpt2kon(mfpt_sample) for mfpt_sample in mfpt_samples]
    l, r = confidence_interval(kon_samples, conf=conf)
    return kon, l, r


def mfpt2koff(mfpt):
    mfpt *= 1e-9  # in seconds
    k_off = 1./mfpt
    return k_off


def unbinding_rate(bhmm, i_micro_unbound, conf=0.95):
    i_unbound, i_bound = index_unbound_bound(bhmm, i_micro_unbound)
    # MLE
    mfpt_mle = bhmm.mfpt(i_bound, i_unbound)
    koff = mfpt2koff(mfpt_mle)
    # samples
    mfpt_samples = bhmm.sample_f('mfpt', i_bound, i_unbound)
    koff_samples = [mfpt2koff(mfpt_sample) for mfpt_sample in mfpt_samples]
    l, r = confidence_interval(koff_samples, conf=conf)
    return koff, l, r