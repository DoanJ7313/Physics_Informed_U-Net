# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_physical_grid(L, W, nx, ny):
    x = np.linspace(0, L, nx)
    y = np.linspace(-W/2, W/2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij') 
    return X, Y

def plot_displacement_contours(u_grid, L, W):
    if isinstance(u_grid, torch.Tensor):
        u_grid = u_grid.cpu().numpy()

    u1 = u_grid[0]  
    u2 = u_grid[1]

    nx, ny = u1.shape
    X, Y = get_physical_grid(L, W, nx, ny)
    
    plt.figure(figsize=(8, 10))
    plt.subplot(2, 1, 1)
    plt.contourf(X, Y, u1, 20, cmap='viridis')
    plt.colorbar()
    plt.title("Displacement - $u_1$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis('equal')
    
    plt.subplot(2, 1, 2)
    plt.contourf(X, Y, u2, 20, cmap='viridis')
    plt.colorbar()
    plt.title("Displacement - $u_2$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def plot_deformed_shape(u_grid, L, W, scale=5.0):
    if isinstance(u_grid, torch.Tensor):
        u_grid = u_grid.cpu().numpy()
        
    u1 = u_grid[0]
    u2 = u_grid[1]
    nx, ny = u1.shape

    X, Y = get_physical_grid(L, W, nx, ny)
    X_def = X + scale * u1
    Y_def = Y + scale * u2

    plt.figure(figsize=(8,6))

    for i in range(nx):
        if i == 0:
            plt.plot(X[i, :], Y[i, :], 'k:', label='undeformed')
            plt.plot(X_def[i, :], Y_def[i, :], 'b-', label='deformed')
        else:
            plt.plot(X[i, :], Y[i, :], 'k:')
            plt.plot(X_def[i, :], Y_def[i, :], 'b-')

    for j in range(ny):
        plt.plot(X[:, j], Y[:, j], 'k:')
        plt.plot(X_def[:, j], Y_def[:, j], 'b-')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis('equal')
    plt.title("Deformed Shape")
    plt.legend()
    plt.show()
    
# Comparisons
plt.rcParams.update({
    "font.size": 16,        # base font size
    "axes.titlesize": 18,   # subplot titles
    "axes.labelsize": 18,   # axis labels
    "legend.fontsize": 16,  # legend
    "xtick.labelsize": 14,  # x tick labels
    "ytick.labelsize": 14   # y tick labels
})


def plot_pred_vs_target(u_pred, u_target, L, W, model_name, cmap='viridis'):
    """
    Compare predicted and target displacement fields side by side.
    u_pred, u_target: [2, nx, ny] tensors or numpy arrays
    """
    # Convert to numpy if tensors
    if isinstance(u_pred, torch.Tensor):
        u_pred = u_pred.cpu().numpy()
    if isinstance(u_target, torch.Tensor):
        u_target = u_target.cpu().numpy()
    
    u1_pred, u2_pred = u_pred[0], u_pred[1]
    u1_target, u2_target = u_target[0], u_target[1]
    
    nx, ny = u1_pred.shape
    X, Y = get_physical_grid(L, W, nx, ny)
    
    # Plot u1
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, u1_target, 20, cmap=cmap)
    plt.colorbar()
    plt.title("Target $u_1$")
    plt.axis('equal')
    
    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, u1_pred, 20, cmap=cmap)
    plt.colorbar()
    plt.title(model_name+ " Predicted $u_1$")
    plt.axis('equal')
    
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, np.abs(u1_pred - u1_target), 20, cmap='coolwarm')
    plt.colorbar()
    plt.title("Error $u_1$ (Pred - Target)")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Plot u2
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, u2_target, 20, cmap=cmap)
    plt.colorbar()
    plt.title("Target $u_2$")
    plt.axis('equal')
    
    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, u2_pred, 20, cmap=cmap)
    plt.colorbar()
    plt.title(model_name+ " Predicted $u_2$")
    plt.axis('equal')
    
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, np.abs(u2_pred - u2_target), 20, cmap='coolwarm')
    plt.colorbar()
    plt.title("Error $u_2$ (Pred - Target)")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def plot_deformed_comparison(u_pred, u_target, L, W, model_name, scale=5.0):
    """
    Overlay predicted and target deformed shapes.
    Blue = predicted, Red = target
    """
    if isinstance(u_pred, torch.Tensor):
        u_pred = u_pred.cpu().numpy()
    if isinstance(u_target, torch.Tensor):
        u_target = u_target.cpu().numpy()
        
    u1_pred, u2_pred = u_pred[0], u_pred[1]
    u1_target, u2_target = u_target[0], u_target[1]
    nx, ny = u1_pred.shape

    X, Y = get_physical_grid(L, W, nx, ny)
    X_def_pred = X + scale * u1_pred
    Y_def_pred = Y + scale * u2_pred
    X_def_target = X + scale * u1_target
    Y_def_target = Y + scale * u2_target

    plt.figure(figsize=(8,6))

    for i in range(nx):
        plt.plot(X[i, :], Y[i, :], 'k:', alpha=0.5)
        plt.plot(X_def_pred[i, :], Y_def_pred[i, :], 'b-', label=model_name + ' Predicted' if i==0 else "")
        plt.plot(X_def_target[i, :], Y_def_target[i, :], 'r--', label='Target' if i==0 else "")

    for j in range(ny):
        plt.plot(X[:, j], Y[:, j], 'k:', alpha=0.5)
        plt.plot(X_def_pred[:, j], Y_def_pred[:, j], 'b-')
        plt.plot(X_def_target[:, j], Y_def_target[:, j], 'r--')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis('equal')
    plt.title("Deformed Shape Comparison")
    plt.legend()
    plt.show()