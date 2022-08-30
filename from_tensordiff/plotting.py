# Raissi et al plotting scripts - https://github.com/maziarraissi/PINNs/blob/master/Utilities/plotting.py
# All code in this script is credited to Raissi et al


import matplotlib as mpl
import numpy as np
import tensorflow as tf
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


import matplotlib.pyplot as plt

def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def plot_solution_domain1D(model, domain, ub, lb, Exact_u=None, u_transpose=False):
    """
    Plot a 1D solution Domain
    Arguments
    ---------
    model : model
        a `model` class which contains the PDE solution
    domain : Domain
        a `Domain` object containing the x,t pairs
    ub: list
        a list of floats containing the upper boundaries of the plot
    lb : list
        a list of floats containing the lower boundaries of the plot
    Exact_u : list
        a list of the exact values of the solution for comparison
    u_transpose : Boolean
        a `bool` describing whether or not to transpose the solution plot of the domain
    Returns
    -------
    None
    """
    X, T = np.meshgrid(domain[0],domain[1])

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    if Exact_u is not None:
        u_star= Exact_u.T.flatten()[:,None]

    u_pred, f_u_pred = model.predict(X_star)
    if u_transpose:
        U_pred = griddata(X_star, u_pred.T.flatten(), (X, T), method='cubic')
    else:
        U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    fig, ax = newfig(1.3, 1.0)

    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    len_ = len(domain[1])//4

    line = np.linspace(domain[0].min(), domain[0].max(), 2)[:,None]
    ax.plot(domain[1][len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(domain[1][2*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(domain[1][3*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    leg = ax.legend(frameon=False, loc = 'best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title(r'$\hat{w}$(t,x)', fontsize = 10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    print("Exact_u",Exact_u.shape,"domain",domain[0].shape,U_pred.shape)
    # print("Exact_u1",Exact_u[:,len_],len_)
    ax.plot(domain[0],Exact_u[:,len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('w(t,x)')
    ax.set_title('t = %.2f' % (domain[1][len_]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    # print("Exact_2",Exact_u[2*len_,:],2*len_)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(domain[0],Exact_u[:,2*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[2*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('w(t,x)')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('t = %.2f' % (domain[1][2*len_]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    # print("exact_3",Exact_u[:,3*len_],3*len_)
    ax.plot(domain[0],Exact_u[:,3*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[3*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('w(t,x)')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('t = %.2f' % (domain[1][3*len_]), fontsize = 10)
    plt.savefig("beam_pinn.png")
    plt.show()


def plot_weights(model, scale = 1):
    plt.scatter(model.domain.X_f[:,1], model.domain.X_f[:,0], c = model.lambdas[0].numpy(), s = model.lambdas[0].numpy()/float(scale))
    plt.xlabel(model.domain.domain_ids[1])
    plt.ylabel(model.domain.domain_ids[0])
    plt.show()
# def plot_weights_(model, scale = 1):
#     plt.scatter(model.x_f, model.t_f, c = model.lambdas[0].numpy(), s = model.lambdas[0].numpy()/float(scale))
#     plt.xlabel("x")
#     plt.ylabel("t")
#     plt.show()

# def plot_glam_values_(model, scale = 1):
#     plt.scatter(model.t_f, model.x_f, c = model.g(model.col_weights).numpy(), s = model.g(model.col_weights).numpy()/float(scale))
#     plt.show()
def plot_glam_values(model, scale = 1):
    plt.scatter(model.domain.X_f[:,1], model.domain.X_f[:,1], c = model.g(model.lambdas[0]).numpy(), s = model.g(model.lambdas[0]).numpy()/float(scale))
    plt.show()
def plot_residuals(FU_pred,ub, lb):
    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(2)
    ec = plt.imshow(FU_pred.T, interpolation='nearest', cmap='rainbow',
                extent=[lb[1], ub[1], lb[0], ub[0]],
                origin='lower', aspect='auto')
    
    #ax.add_collection(ec)
    ax.autoscale_view()
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    cbar = plt.colorbar(ec)
    cbar.set_label('$\overline{f}_u$ prediction')
    plt.savefig("pinn_residual.png")
    plt.show()

def get_griddata(grid, data, dims):
    return griddata(grid, data, dims, method='cubic')

def plot(x,t,z,ub, lb):
    # x_plot =tf.squeeze(x,[1])
    # t_plot =tf.squeeze(t,[1])
    X,T= np.meshgrid(x,t)
    F_xt = z
    print(X.shape,T.shape,F_xt.shape)
    fig,ax=newfig(1.3, 1.0)
    ax.axis("off")
    
    ##########
    fig.set_figwidth(8)
    fig.set_figheight(2)
    
    gs0 = gridspec.GridSpec(1, 2)
    # gs0.update(top=1-1/4, bottom=1-1/2, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    cp = ax.contourf(X,T, F_xt,30,cmap="YlGnBu",extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    len_ = len(t)//3
    print(len(t))
    ax.plot(t[len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[2*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    # ax.plot(t[2*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cp,cax) # Add a colorbar to a plot
    ax.set_title('w(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ###########
    # gs1 = gridspec.GridSpec(1, 2)
    # gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    # ax = plt.subplot(gs0[2,:])
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(T, X, F_xt,cmap="YlGnBu")
    # ax.set_xlabel('t')
    # ax.set_ylabel('x')
    # ax.set_zlabel('w(x,t)')
    plt.savefig("original_beam_pinn.png")
    plt.show()
def plot_discovery(model, scale = 1):
    plt.scatter(model.domain.X_f[:,1], model.domain.X_f[:,0])
    plt.xlabel(model.domain.domain_ids[1])
    plt.ylabel(model.domain.domain_ids[0])
    plt.show()
