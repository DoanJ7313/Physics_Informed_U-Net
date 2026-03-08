import torch
from global_var import config

def energy_loss(u_h, u_true, D, t, L=48, W=20, beta=100):
    """
    Inputs:
        u_h: [B, 2, nx, ny] predicted displacement
        u_true: [B, 2, nx, ny] true displacement
        t: [B, 2, nx, ny] Neumann traction
        D: [3, 3] material stiffness matrix
        L, W: physical domain dimensions
        beta: Nitsche penalty for Dirichlet BC
    Outputs:
        scalar loss
    """
    BATCH, C, NX, NY = u_h.shape
    device = u_h.device
    D = D.to(device)
    n_nodes = NX * NY

    dx = L / (NX - 1)
    dy = W / (NY - 1)

    # Compute finite difference gradients (central diff)
    u_x = torch.zeros_like(u_h)
    u_y = torch.zeros_like(u_h)
    
    u_x[:, :, 1:-1, :] = (u_h[:, :, 2:, :] - u_h[:, :, :-2, :]) / (2*dx)
    u_y[:, :, :, 1:-1] = (u_h[:, :, :, 2:] - u_h[:, :, :, :-2]) / (2*dy)

    # Boundaries (forward/backward)
    u_x[:, :, 0, :] = (u_h[:, :, 1, :] - u_h[:, :, 0, :]) / dx
    u_x[:, :, -1, :] = (u_h[:, :, -1, :] - u_h[:, :, -2, :]) / dx

    u_y[:, :, :, 0] = (u_h[:, :, :, 1] - u_h[:, :, :, 0]) / dy
    u_y[:, :, :, -1] = (u_h[:, :, :, -1] - u_h[:, :, :, -2]) / dy

    # Strain components 
    exx = u_x[:, 0, :, :]          
    eyy = u_y[:, 1, :, :]           
    exy = 0.5 * (u_x[:, 1, :, :] + u_y[:, 0, :, :])  
    exx[:, -1, :] = (u_h[:, 0, -1, :] - u_h[:, 0, -2, :]) / dx
    exy[:, -1, :] = 0.5 * ((u_h[:, 1, -1, :] - u_h[:, 1, -2, :])/dx + (u_h[:, 0, -1, :] - u_h[:, 0, -2, :])/dy)
    exx[:, -2, :] = (u_h[:, 0, -2, :] - u_h[:, 0, -3, :]) / dx
    exy[:, -2, :] = 0.5 * (
    (u_h[:, 1, -2, :] - u_h[:, 1, -3, :]) / dx + 
    (u_h[:, 0, -2, :] - u_h[:, 0, -3, :]) / dy)

    # Stack strain tensor
    strain = torch.stack([exx, eyy, exy], dim=1)  # [B,3,NX,NY]

    # Stress
    stress = torch.einsum('ij,bjxy->bixy', D, strain)  # [B,3,NX,NY]

    # Domain energy: 0.5 * sigma : epsilon * dx*dy
    energy_density = 0.5 * (strain * stress).sum(dim=1)  # [B,NX,NY]
    term1 = energy_density.sum(dim=(1,2)) * dx * dy
    
    # Neumann BC
    u_h_neum = u_h[:, :, -1, :]     
    t_neum = t[:, :, -1, :]      
    term2 = -(t_neum * u_h_neum).sum(dim=(1,2)) * dy

    # Dirichlet BC
    u_h_diric = u_h[:, :, 0, :]
    u_diric = u_true[:, :, 0, :]
    u_diff = u_h_diric - u_diric

    sigma_xx = stress[:, 0, 0, :]
    sigma_xy = stress[:, 2, 0, :]
    
    t_x = -sigma_xx
    t_y = -sigma_xy
    
    traction_diric = torch.stack([t_x, t_y], dim=1)

    term3 = -(traction_diric * u_diff).sum(dim=(1,2)) * dy

    # Penalty
    term4 = 0.5 * beta * (u_diff**2).sum(dim=(1,2)) * dy

    energy = term1 + term2 + term3 + term4

    # True solution internal energy (normalization scale)
    with torch.no_grad():

        u_x_true = torch.zeros_like(u_true)
        u_y_true = torch.zeros_like(u_true)

        u_x_true[:, :, 1:-1, :] = (u_true[:, :, 2:, :] - u_true[:, :, :-2, :]) / (2*dx)
        u_y_true[:, :, :, 1:-1] = (u_true[:, :, :, 2:] - u_true[:, :, :, :-2]) / (2*dy)

        u_x_true[:, :, 0, :] = (u_true[:, :, 1, :] - u_true[:, :, 0, :]) / dx
        u_x_true[:, :, -1, :] = (u_true[:, :, -1, :] - u_true[:, :, -2, :]) / dx

        u_y_true[:, :, :, 0] = (u_true[:, :, :, 1] - u_true[:, :, :, 0]) / dy
        u_y_true[:, :, :, -1] = (u_true[:, :, :, -1] - u_true[:, :, :, -2]) / dy

        exx_t = u_x_true[:, 0]
        eyy_t = u_y_true[:, 1]
        exy_t = 0.5 * (u_x_true[:, 1] + u_y_true[:, 0])

        strain_true = torch.stack([exx_t, eyy_t, exy_t], dim=1)

        stress_true = torch.einsum('ij,bjxy->bixy', D, strain_true)

        energy_density_true = 0.5 * (strain_true * stress_true).sum(dim=1)

        true_term1 = energy_density_true.sum(dim=(1,2)) * dx * dy

        u_true_neum = u_true[:, :, -1, :]        
        true_term2 = -(t_neum * u_true_neum).sum(dim=(1,2)) * dy

        true_energy = true_term1 + true_term2 

    # Normalize by true energy
    energy = torch.abs(energy - true_energy) / (torch.abs(true_energy).clamp(min=1e-4))

    loss = torch.abs(energy).mean()
    return loss

def test_energy_loss(u_h, u_true, D, t, L=48, W=20, beta=100):
    """
    Inputs:
        u_h: [B, 2, nx, ny] predicted displacement
        u_true: [B, 2, nx, ny] true displacement
        t: [B, 2, nx, ny] Neumann traction
        D: [3, 3] material stiffness matrix
        L, W: physical domain dimensions
        beta: Nitsche penalty for Dirichlet BC
    Outputs:
        scalar loss
    """
    BATCH, C, NX, NY = u_h.shape
    device = u_h.device
    D = D.to(device)
    n_nodes = NX * NY

    dx = L / (NX - 1)
    dy = W / (NY - 1)

    # Compute finite difference gradients (central diff)
    u_x = torch.zeros_like(u_h)
    u_y = torch.zeros_like(u_h)
    
    u_x[:, :, 1:-1, :] = (u_h[:, :, 2:, :] - u_h[:, :, :-2, :]) / (2*dx)
    u_y[:, :, :, 1:-1] = (u_h[:, :, :, 2:] - u_h[:, :, :, :-2]) / (2*dy)

    # Boundaries (forward/backward)
    u_x[:, :, 0, :] = (u_h[:, :, 1, :] - u_h[:, :, 0, :]) / dx
    u_x[:, :, -1, :] = (u_h[:, :, -1, :] - u_h[:, :, -2, :]) / dx

    u_y[:, :, :, 0] = (u_h[:, :, :, 1] - u_h[:, :, :, 0]) / dy
    u_y[:, :, :, -1] = (u_h[:, :, :, -1] - u_h[:, :, :, -2]) / dy

    # Strain components 
    exx = u_x[:, 0, :, :]          
    eyy = u_y[:, 1, :, :]           
    exy = 0.5 * (u_x[:, 1, :, :] + u_y[:, 0, :, :])  
    exx[:, -1, :] = (u_h[:, 0, -1, :] - u_h[:, 0, -2, :]) / dx
    exy[:, -1, :] = 0.5 * ((u_h[:, 1, -1, :] - u_h[:, 1, -2, :])/dx + (u_h[:, 0, -1, :] - u_h[:, 0, -2, :])/dy)
    exx[:, -2, :] = (u_h[:, 0, -2, :] - u_h[:, 0, -3, :]) / dx
    exy[:, -2, :] = 0.5 * (
    (u_h[:, 1, -2, :] - u_h[:, 1, -3, :]) / dx + 
    (u_h[:, 0, -2, :] - u_h[:, 0, -3, :]) / dy)

    # Stack strain tensor
    strain = torch.stack([exx, eyy, exy], dim=1)  # [B,3,NX,NY]

    # Stress
    stress = torch.einsum('ij,bjxy->bixy', D, strain)  # [B,3,NX,NY]

    # Domain energy: 0.5 * sigma : epsilon * dx*dy
    energy_density = 0.5 * (strain * stress).sum(dim=1)  # [B,NX,NY]
    term1 = energy_density.sum(dim=(1,2)) * dx * dy  # [B]
    
    # Neumann BC
    u_h_neum = u_h[:, :, -1, :]     
    t_neum = t[:, :, -1, :]      
    term2 = -(t_neum * u_h_neum).sum(dim=(1,2)) * dy

    # Dirichlet BC
    u_h_diric = u_h[:, :, 0, :]
    u_diric = u_true[:, :, 0, :]
    u_diff = u_h_diric - u_diric

    sigma_xx = stress[:, 0, 0, :]
    sigma_xy = stress[:, 2, 0, :]
    
    t_x = -sigma_xx
    t_y = -sigma_xy
    
    traction_diric = torch.stack([t_x, t_y], dim=1)

    term3 = -(traction_diric * u_diff).sum(dim=(1,2)) * dy

    # Penalty
    # beta_eff = beta / config["weight_en"]
    term4 = 0.5 * beta * (u_diff**2).sum(dim=(1,2)) * dy

    energy = term1 + term2 + term3 + term4

    # True solution internal energy (normalization scale)
    with torch.no_grad():

        u_x_true = torch.zeros_like(u_true)
        u_y_true = torch.zeros_like(u_true)

        u_x_true[:, :, 1:-1, :] = (u_true[:, :, 2:, :] - u_true[:, :, :-2, :]) / (2*dx)
        u_y_true[:, :, :, 1:-1] = (u_true[:, :, :, 2:] - u_true[:, :, :, :-2]) / (2*dy)

        u_x_true[:, :, 0, :] = (u_true[:, :, 1, :] - u_true[:, :, 0, :]) / dx
        u_x_true[:, :, -1, :] = (u_true[:, :, -1, :] - u_true[:, :, -2, :]) / dx

        u_y_true[:, :, :, 0] = (u_true[:, :, :, 1] - u_true[:, :, :, 0]) / dy
        u_y_true[:, :, :, -1] = (u_true[:, :, :, -1] - u_true[:, :, :, -2]) / dy

        exx_t = u_x_true[:, 0]
        eyy_t = u_y_true[:, 1]
        exy_t = 0.5 * (u_x_true[:, 1] + u_y_true[:, 0])

        strain_true = torch.stack([exx_t, eyy_t, exy_t], dim=1)

        stress_true = torch.einsum('ij,bjxy->bixy', D, strain_true)

        energy_density_true = 0.5 * (strain_true * stress_true).sum(dim=1)

        true_term1 = energy_density_true.sum(dim=(1,2)) * dx * dy

        u_true_neum = u_true[:, :, -1, :]        
        true_term2 = -(t_neum * u_true_neum).sum(dim=(1,2)) * dy

        true_energy = true_term1 + true_term2 

    energy = torch.abs(energy - true_energy) / (torch.abs(true_energy).clamp(min=1e-4))

    loss = torch.abs(energy).mean()
    return loss