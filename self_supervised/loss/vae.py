
def kl_divergence(mu, logvar):
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = klds.sum(1).mean(0, True)
    return kld

def i_kl_divergence(mu, logvar, mup, logl):
    klds = -0.5 * (1 + logvar - logl - (mu - mup).pow(2) - logvar.exp() / logl.exp())
    kld = klds.sum(1).mean(0, True)
    return kld

def reconstruction_loss(x_recon, x):
    return ((x_recon - x).pow(2)).mean()

def reconstruction_loss_w_obs_noise(x_recon, x, obs_log_var):
    # error / 2 sigma + sigma/2
    return ((x_recon - x).pow(2) / (2 * obs_log_var.exp()) + obs_log_var/2).mean()

def poisson_reconstruction_loss(x_recon, x):
    return (x_recon - x * x_recon.log()).mean()
