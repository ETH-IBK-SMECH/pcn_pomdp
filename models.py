import pymc3 as pm
import theano.tensor as tt
import numpy as np


# HMM
class HMMStates(pm.Categorical):
    def __init__(self, p_transition, init_prob, actions, n_states, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.p_transition = p_transition
        self.init_prob = init_prob
        self.actions = actions
        self.k = n_states
        self.mode = tt.cast(0,dtype='int64')

    def logp(self, x):
        p_init = self.init_prob
        acts = self.actions[:-1]
        p_tr = self.p_transition[acts, x[:-1]]

        # the logp of the initial state
        initial_state_logp = pm.Categorical.dist(p_init).logp(x[0])

        # the logp of the rest of the states.
        x_i = x[1:]
        ou_like = pm.Categorical.dist(p_tr).logp(x_i)
        transition_logp = tt.sum(ou_like)
        return initial_state_logp + transition_logp


class ARHMMGaussianEmissions(pm.Continuous):
    def __init__(self, states, mu, sigma, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = states
        self.sigma = sigma
        self.mu = mu
        self.k = k

    def logp(self, x):
        """
        x: observations
        """
        states = self.states
        mu = self.mu[states]
        sigma = self.sigma[states]
        k = self.k

        ar_mean = k * x[:-1]
        ar_like = tt.sum(pm.Normal.dist(mu=ar_mean + mu[1:], sigma=sigma[1:]).logp(x[1:]))

        boundary_like = pm.Normal.dist(mu=mu[0], sigma=sigma[0]).logp(x[0])
        return ar_like + boundary_like


class HMMGaussianEmissions(pm.Continuous):
    def __init__(self, states, mu, sigma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = states
        self.sigma = sigma
        self.mu = mu

    def logp(self, x):
        """
        x: observations
        """
        states = self.states
        mu = self.mu[states]
        sigma = self.sigma[states]

        ar_like = tt.sum(pm.Normal.dist(mu=mu[1:], sigma=sigma[1:]).logp(x[1:]))
        boundary_like = pm.Normal.dist(mu=mu[0], sigma=sigma[0]).logp(x[0])
        return boundary_like + ar_like


def define_model(n_states, n_actions, sequences):

    n_nodes = sequences['states'].shape[0]

    with pm.Model() as model:

        min_prior = 0.2
        transition_mat_0 = pm.Dirichlet(
            'p_degradation',
            a=np.full(shape=(n_states, n_states), fill_value=min_prior) + np.diag((7 - min_prior) * np.ones(n_states)) + np.diag((1 - min_prior) * np.ones(n_states - 1), k=1),
            shape=(n_states, n_states)
        )

        transition_mat_1 = pm.Dirichlet(
            'p_repair',
            a=np.full(shape=(n_states, n_states), fill_value=min_prior) + np.tril(np.arange(n_states**2)[::-1].reshape(n_states, n_states).T + 1 - min_prior),
            shape=(n_states, n_states)
        )

        transition_mat = pm.Deterministic("p_transition",
                                          tt.stack([transition_mat_0, transition_mat_1]))

        init_probs = pm.Dirichlet('init_probs', a=tt.ones((n_states,)), shape=n_states)

        # Prior for mu, sigma and k
        mu_estimator = pm.Normal("mu", mu=np.arange(n_states), sigma=0.5,
                                 shape=(n_states,))  # different priors for the state Normal dist
        sigma_estimator = pm.Exponential("sigma", lam=3, shape=(n_states,))

        """
        hmm_states = HMMStates(
                "hmm_states",
                p_transition=transition_mat,
                init_prob=init_probs,
                n_states=n_states,
                actions=sequences['actions'],
                shape=sequences['actions'].shape
            )

        obs = HMMGaussianEmissions(
            "emissions",
            states=hmm_states,
            mu=mu_estimator,
            sigma=sigma_estimator,
            observed=sequences['emissions'],
            testval=sequences['emissions'][~np.isnan(sequences['emissions'])].mean()
        )
        """

        for i in range(n_nodes):
            # HMM state
            hmm_states = HMMStates(
                "hmm_states_{}".format(i),
                p_transition=transition_mat,
                init_prob=init_probs,
                n_states=n_states,
                actions=sequences['actions'][i],
                shape=(len(sequences['actions'][i]),)
            )

            # Observed emission likelihood
            obs = HMMGaussianEmissions(
                "emission_{}".format(i),
                states=hmm_states,
                mu=mu_estimator,
                sigma=sigma_estimator,
                observed=sequences['emissions'][i],
                testval=sequences['emissions'][i][~np.isnan(sequences['emissions'][i])].mean()
            )
    return model
