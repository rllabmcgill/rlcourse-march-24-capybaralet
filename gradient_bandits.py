"""
We evaluate the gradient bandit algorithm over several environments, studying the impact of baselines.

Note that the order of the plots is different than in the write-up
"""

#----------------------------------------
# imports and functions
import numpy
np = numpy
import numpy.random
from pylab import *

def softmax(vec):
    """
    A 1d softmax for a single vector input
    """
    mm = np.max(vec)
    return np.exp(vec - mm) / np.sum(np.exp(vec - mm))

def update_mean(mean, obs, num_observed):
    return 1. / num_observed * ((num_observed-1) * mean + obs)



#----------------------------------------
# BANDIT ENVIRONMENTS
#   We use the 10-arm testbed, as described in Sutton Barto 2017, but shift the means by some varying amount
#   Bandits are encoded as the mean of the arms.
num_envs = 2000
testbed = np.random.normal(size=(num_envs,10))
optimal_actions = np.argmax(testbed, axis=1)

#----------------------------------------
# BASELINES:
baseline_names = [
        'none',
        'avg_reward',
        'V_pi',
        'V_star',
        'model_based']
gammas = [.5, .9, .99, .999, .9999]
baseline_names += ['gamma=' + str(gamma) for gamma in gammas]


#----------------------------------------
# RUN 

mean_shifts = [0,2,4,8]
num_steps = 1000
alpha = 0.1 # learning rate

# logging
rewards = np.empty(( len(mean_shifts), len(testbed), len(baseline_names), num_steps )) # we center the rewards for ease of comparison (i.e. subtract the shift)
actions = np.empty(( len(mean_shifts), len(testbed), len(baseline_names), num_steps ))
optimal = np.empty(( len(mean_shifts), len(testbed), len(baseline_names), num_steps ))
baseline_error = np.empty(( len(mean_shifts), len(testbed), len(baseline_names), num_steps ))
baselines = np.empty(( len(mean_shifts), len(testbed), len(baseline_names), num_steps ))
P_opt = np.empty(( len(mean_shifts), len(testbed), len(baseline_names), num_steps ))


import time
t0 = time.time()
for shift_n, shift in enumerate(mean_shifts):

    # apply mean shift
    testbed_ = testbed + shift

    for bandit_n, bandit in enumerate(testbed_):

        # compute the true value function of this bandit
        V_star = np.max(bandit)
        optimal_action = np.argmax(bandit)

        for baseline_n, baseline_name in enumerate(baseline_names):

            # re-initialize policy and baseline, etc.
            preferences = np.zeros(10)
            baseline = 0
            unnormalized_baseline = 0
            R_t = 0
            reward_means = np.zeros(10)
            num_reward_obs = np.zeros(10)

            for step in range(num_steps):

                policy = softmax(preferences)

                # SAMPLE ACTION AND REWARD
                A_t = np.argmax(np.random.multinomial(n=1, pvals=policy))
                R_t = numpy.random.randn() + bandit[A_t]
                # logging
                rewards[shift_n, bandit_n, baseline_n, step] = R_t - shift
                #actions[shift_n, bandit_n, baseline_n, step] = A_t
                optimal[shift_n, bandit_n, baseline_n, step] = (A_t == optimal_action)
                baseline_error[shift_n, bandit_n, baseline_n, step] = (baseline - np.sum(policy * bandit))**2
                #baselines[shift_n, bandit_n, baseline_n, step] = baseline 
                P_opt[shift_n, bandit_n, baseline_n, step] = policy[optimal_action]

                # COMPUTE BASELINE
                if baseline_name == 'none':
                    baseline = 0
                elif baseline_name == 'avg_reward':
                    baseline = update_mean(baseline, R_t, step+1) # +1 to avoid divide by 0
                elif baseline_name == 'V_pi':
                    baseline = np.sum(policy * bandit)
                elif baseline_name == 'V_star':
                    baseline = V_star
                elif baseline_name == 'model_based':
                    num_reward_obs[A_t] += 1
                    reward_means[A_t] = update_mean(reward_means[A_t], R_t, num_reward_obs[A_t])
                    baseline = np.sum(policy * reward_means)
                else: # exponential moving average
                    gamma = float(baseline_name.split('gamma=')[1])
                    unnormalized_baseline = (gamma * unnormalized_baseline + R_t)
                    baseline = unnormalized_baseline * (1. - gamma) / (1.- gamma**(step+1))

                # APPLY UPDATE
                for action in range(10):
                    if action == A_t:
                        preferences[action] += alpha * (R_t - baseline) * (1 - policy[A_t])
                    else:
                        preferences[action] -= alpha * (R_t - baseline) * policy[A_t]
                

print num_envs, "envs    ", (time.time() - t0) / num_envs, "seconds per env"

np.save('rewards.npy', rewards)
np.save('optimal.npy', optimal)
np.save('baseline_error.npy', baseline_error)
np.save('P_opt.npy', P_opt)


#----------------------------------------------------
# PLOT RESULTS

def averaged(results):
    return np.cumsum(results, axis=1) / np.arange(results.shape[1]).reshape((1,-1))

def err_plot(samples, **kwargs):
    """
    plot averaged samples with (standard error) error-bars
    samples shape: (sample_n, ___)
    """
    means = samples.mean(axis=0)
    stds = samples.std(axis=0) / samples.shape[0]**.5
    plt.errorbar(range(samples.shape[1]), means, stds, **kwargs) 
    #plt.plot(means, **kwargs)



# -------------------------------
# PLOTS: average reward and % correct actions taken
# -------------------------------

# FIGURE 1: Compare all approaches
figure()
suptitle('performance of different baselines')

for shift_n in range(4):

    # select best gamma (based on rewards)
    best_gamma_ind = np.argmax([np.mean(averaged(rewards[shift_n, :, nn+5])[:, -1]) for nn, _ in enumerate(gammas)]) + 5

    # rewards
    subplot(4,2,2*shift_n+1)
    title("average mean of arms shifted to ~" + str(mean_shifts[shift_n]))
    for nn, name in enumerate(baseline_names[:5]):
        err_plot(averaged(rewards[shift_n, :, nn]), label=name)
        ylabel('average R_t')
        ylim(.8,1.5)
    err_plot(averaged(rewards[shift_n, :, best_gamma_ind]), label=baseline_names[best_gamma_ind])

    # optimal
    subplot(4,2,2*shift_n+2)
    title("average mean of arms shifted to ~" + str(mean_shifts[shift_n]))
    for nn, name in enumerate(baseline_names[:5]):
        err_plot(averaged(optimal[shift_n, :, nn]), label=name)
        ylabel('% optimal action')
    err_plot(averaged(optimal[shift_n, :, best_gamma_ind]), label=baseline_names[best_gamma_ind])

    legend()
subplot(4,2,7)
xlabel('step')
subplot(4,2,8)
xlabel('step')
subplots_adjust(hspace=0.4)
get_current_fig_manager().window.showMaximized()
savefig('fig1.pdf')



# FIGURE 2: Compare different exponential discounts
figure()
suptitle(r'performance of different $\gamma$s')

for shift_n in range(4):

    # rewards
    subplot(4,2,2*shift_n+1)
    title("average mean of arms shifted to ~" + str(mean_shifts[shift_n]))
    for nn, gamma in enumerate(gammas):
        err_plot(averaged(rewards[shift_n, :, nn+5]), label=r'$\gamma='+str(gamma))
        ylabel('average R_t')
        ylim(.8,1.5)

    # optimal
    subplot(4,2,2*shift_n+2)
    title("average mean of arms shifted to ~" + str(mean_shifts[shift_n]))
    for nn, gamma in enumerate(gammas):
        err_plot(averaged(optimal[shift_n, :, nn+5]), label=r'$\gamma='+str(gamma))
        ylabel('% optimal action')

subplot(4,2,7)
xlabel('step')
subplot(4,2,8)
xlabel('step')
legend()
subplots_adjust(hspace=0.4)
get_current_fig_manager().window.showMaximized()
savefig('fig2.pdf')








# -------------------------------
# PLOTS: baseline error and pi(a*)
# -------------------------------


# FIGURE 3: Compare all approaches
figure()
suptitle('performance of different baselines')

for shift_n in range(4):

    # select best gamma (based on rewards)
    best_gamma_ind = np.argmax([np.mean(averaged(rewards[shift_n, :, nn+5])[:, -1]) for nn, _ in enumerate(gammas)]) + 5

    # baseline error
    subplot(4,2,2*shift_n+1)
    title("average mean of arms shifted to ~" + str(mean_shifts[shift_n]))
    for nn, name in enumerate(baseline_names[:5]):
        err_plot(baseline_error[shift_n, :, nn], label=name)
        ylabel('MSE of baseline')
        ylim(-.05,.75)
    err_plot(baseline_error[shift_n, :, best_gamma_ind], label=baseline_names[best_gamma_ind])

    # pi(a*)
    subplot(4,2,2*shift_n+2)
    title("average mean of arms shifted to ~" + str(mean_shifts[shift_n]))
    for nn, name in enumerate(baseline_names[:5]):
        err_plot(P_opt[shift_n, :, nn], label=name)
        ylabel('pi(a*)')
    err_plot(P_opt[shift_n, :, best_gamma_ind], label=baseline_names[best_gamma_ind])

    legend()
subplot(4,2,7)
xlabel('step')
subplot(4,2,8)
xlabel('step')
subplots_adjust(hspace=0.4)
get_current_fig_manager().window.showMaximized()
savefig('fig3.pdf')



# FIGURE 4: Compare different exponential discounts
figure()
suptitle(r'performance of different $\gamma$s')

for shift_n in range(4):

    # baseline error
    subplot(4,2,2*shift_n+1)
    title("average mean of arms shifted to ~" + str(mean_shifts[shift_n]))
    for nn, gamma in enumerate(gammas):
        err_plot(baseline_error[shift_n, :, nn+5], label=r'$\gamma='+str(gamma))
        ylabel('MSE of baseline')
        ylim(-.05,.75)

    # pi(a*)
    subplot(4,2,2*shift_n+2)
    title("average mean of arms shifted to ~" + str(mean_shifts[shift_n]))
    for nn, gamma in enumerate(gammas):
        err_plot(P_opt[shift_n, :, nn+5], label=r'$\gamma='+str(gamma))
        ylabel('pi(a*)')

subplot(4,2,7)
xlabel('step')
subplot(4,2,8)
xlabel('step')
legend()
subplots_adjust(hspace=0.4)
get_current_fig_manager().window.showMaximized()
savefig('fig4.pdf')




