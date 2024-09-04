import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

# WIP
predefined_hamiltonian = {1:[[qml.PauliZ(0)]],
                          2:[[ qml.PauliZ(0) @ qml.PauliZ(1) ], # Entangler
                             [ qml.PauliZ(0), qml.PauliZ(1) ]], # Phase accumulator
                          3:[[ qml.PauliZ(0) @ qml.PauliZ(1) @ qml.Identity(2), # Entangler
                               qml.Identity(0) @ qml.PauliZ(1) @ qml.PauliZ(2),
                               qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)],
                             [ qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)]], # Phase accumulator
                          4:[[ qml.PauliZ(0) @ qml.PauliZ(1) @ qml.Identity(2) @ qml.Identity(3), # Entangler
                               qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2) @ qml.Identity(3),
                               qml.PauliZ(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.PauliZ(3),
                               qml.Identity(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.Identity(3),
                               qml.Identity(0) @ qml.PauliZ(1) @ qml.Identity(2) @ qml.PauliZ(3),
                               qml.Identity(0) @ qml.Identity(1) @ qml.PauliZ(2) @ qml.PauliZ(3)],
                             [ qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2), qml.PauliZ(3)]]} #Phase Accumulator

num_coeffs = [1,1,3,6]
num_params = [2,6,8,16]

visualization_layout = {1:(1,3), 2:(3,3), 3:(3,4), 4:(4,5)}

# WIP
data_label = {1:[r'$\theta_{RZ}$', r'$\theta_{RX}$'], 
              
              2:[r'$\theta_{RX_{1}}$', r'$\theta_{RX_{2}}$',
                 r'$\theta_{RZ_{1}}$', r'$\theta_{RZ_{2}}$',
                 r'$Entangle_{1}$', r'$Entangle_{2}$'], 

              3:[r'$\theta_{RX_{1}}$', r'$\theta_{RX_{2}}$', r'$\theta_{RX_{3}}$',
                 r'$\theta_{RZ_{1}}$', r'$\theta_{RZ_{2}}$', r'$\theta_{RZ_{3}}$',
                 r'$Entangle_{1}$', r'$Entangle_{2}$'],  

              4:[r'$\theta_{RX_{(1,1)}}$', r'$\theta_{RX_{(1,2)}}$', r'$\theta_{RX_{(1,3)}}$', r'$\theta_{RX_{(1,4)}}$',
                 r'$\theta_{RX_{(2,1)}}$', r'$\theta_{RX_{(2,2)}}$', r'$\theta_{RX_{(2,3)}}$', r'$\theta_{RX_{(2,4)}}$',
                 r'$\theta_{RZ_{1}}$', r'$\theta_{RZ_{2}}$', r'$\theta_{RZ_{3}}$', r'$\theta_{RZ_{4}}$',
                 r'$Entangle_{(1,1)}$', r'$Entangle_{(1,2)}$', r'$Entangle_{(2,1)}$', r'$Entangle_{(2,2)}$']
            }

def dephase_factor(tau):
    """ 
    Calculate the dephasing factor for a given dephasing time tau.

    Args:
        tau (torch.Tensor): Dephasing time.

    Returns:
        torch.Tensor: Dephasing factor.
    """
    if isinstance(tau, float):
        return 1 - torch.exp(torch.tensor(-2 * tau))
    elif isinstance(tau, torch.Tensor):
        return 1 - torch.exp(-2*tau.clone().detach().requires_grad_(True))
    else:
        raise TypeError('Invalid type of tau')


def sweep_range(t_obs: float, num_points: int):
    """
    Returns the sweep range for optimization.

    Args:
        start (float): starting time point.
        end (float): ending time point.
        num_points(int): number of time points to be optimized.
    """
    return torch.tensor(np.linspace(0.0, t_obs, num_points))


def plot_data(data, num_qubit, k, freq, gamma, filename):

    timespace = data[:,0]

    reference = (num_qubit**2)*np.exp(-2*timespace*k*freq)
    layout = visualization_layout[num_qubit]
    plt.figure(figsize=(12,4))
    y = data[:,1]

    plt.subplot(layout[0], layout[1], 1)
    plt.title('CFI')
    plt.plot(timespace, y, label='CFI')
    plt.plot(timespace, reference, label= 'Reference', linestyle='dotted')
    plt.ylim(0,max(y))
    plt.legend()

    if num_qubit == 1:
        for i in range(2):
            plt.subplot(layout[0], layout[1], i+2)
            plt.title(data_label[num_qubit][i])
            plt.yticks(
                [-2*pi, -3*pi/2, 0, -pi, -pi/2, 0, pi/2, pi, 3*pi/2, 2*pi], 
                [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
            )
            plt.ylim(-2*pi, 2*pi)
            plt.plot(timespace, data[:,i+2])

    elif num_qubit <= 4:

        pivot, entangle = None, None

        if num_qubit < 4:
            pivot = 2
            entangle = 2
        else:
            pivot = 3
            entangle = 4

        for i in range(pivot):
            for j in range(num_qubit):
                plt.subplot(layout[0], layout[1], layout[1]*i+j+2)
                plt.title(data_label[num_qubit][i*num_qubit+j])
                plt.yticks(
                    [-2*pi, -3*pi/2, 0, -pi, -pi/2, 0, pi/2, pi, 3*pi/2, 2*pi], 
                    [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
                )
                plt.ylim(-2*pi, 2*pi)
                plt.plot(timespace, data[:,i*num_qubit+j+2])
        for i in range(entangle):
            plt.subplot(layout[0], layout[1], layout[1]*pivot+i+2)
            plt.title(data_label[num_qubit][pivot*num_qubit+i])
            plt.yticks(
                [-2*pi, -3*pi/2, 0, -pi, -pi/2, 0, pi/2, pi, 3*pi/2, 2*pi], 
                [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
            )
            plt.ylim(-2*pi, 2*pi)
            plt.plot(timespace, data[:,pivot*num_qubit+i+2])
    
    plt.savefig(filename+'.png')
    #plt.show()
