import torch
import pennylane as qml

from utils.utils import *
from utils.arguments import circuitarguments

pi = np.pi
    
class OptimizationCircuit():
    
    def __init__(self, args: circuitarguments):
        """
        Circuit Initializer.

        Args:
            args (circuitarguments) : arguments for initialization.
        """

        self.dev = qml.device('default.mixed', wires=args.num_qubit)

        # Variable settings
        self.num_qubit = args.num_qubit
        self.num_points = args.num_points
        self.dephase_freq = args.freq
        self.t2 = args.t2
        self.ps_gamma = torch.tensor(args.gamma, dtype = torch.float64, requires_grad = False)
        self.ratio = 28.024e+9

        self.no_ps = False


        if self.ps_gamma == 0:
            self.no_ps = True

        template = None
        size = (num_params[self.num_qubit-1],)

        template = 2*pi*(np.random.random(size)-0.5)
        
        if self.num_qubit == 1:
            if self.no_ps:
                template = [0.,0.]
            else:
                template = [pi, pi/2]
        # elif self.num_qubit < 4:
        #     template[-2:] = 0
        # else:
        #     template[-4:] = 0

        self.params = torch.tensor(template, requires_grad=True)

        self.t_obs = args.t_obs
        self.sweep_list = sweep_range(args.t_obs, args.num_points)

        self.obs = predefined_hamiltonian[self.num_qubit]
        self.coeffs = num_coeffs[self.num_qubit-1]

        # Hamiltonian and Kraus operator for post-selection definition
        # self.H = qml.Hamiltonian(
        #             coeffs = [self.dephase_freq/2] * self.coeffs, 
        #             observables = self.obs[0]
        #         )
        
        # self.H_1 = None
        # if self.num_qubit > 1:
        #     self.H_1 = qml.Hamiltonian(
        #         coeffs = [self.dephase_freq/2] * self.num_qubit,
        #         observables = self.obs[1]
        #     )
        
        self.K = torch.tensor([
            [torch.sqrt(1 - self.ps_gamma), 0],
            [0,1]
        ], dtype=torch.complex128)

        tmp = self.K
        for _ in range(self.num_qubit-1):
            self.K = torch.kron(self.K, tmp)

        # Circuit definition
        @qml.qnode(self.dev, interface = 'torch', diff_method = 'backprop')
        def circuit(param: torch.Tensor):
            
            t = param[1].item()
            tau = dephase_factor(t / self.t2)

            H = qml.Hamiltonian(
                    coeffs = [self.ratio*param[0]/2] * self.coeffs, 
                    observables = self.obs[0]
                )
            if self.num_qubit > 1:
                H_1 = qml.Hamiltonian(
                    coeffs = [self.ratio*param[0]/2] * self.num_qubit,
                    observables = self.obs[1]
                )

            # State initialization
            if self.num_qubit == 1:
                qml.RX(pi/2, wires=0)

                # Time evolution
                qml.ApproxTimeEvolution(H, t, 1)
                # Phase damping and rotation by parameters
                qml.PhaseDamping(tau, wires = 0)
                qml.RZ(param[2], wires = 0)
                qml.RX(param[3], wires = 0)

            elif self.num_qubit < 4:
                for i in range(self.num_qubit):
                    qml.RY(pi/2, wires=i)

                # Entangler
                et = torch.abs(param[-3]/param[-1])
                qml.ApproxTimeEvolution(H, et, 1)
                #dp = dephase_factor(et/self.t2)
                for i in range(self.num_qubit):
                    #qml.PhaseDamping(dp, wires=i)
                    qml.RX(param[i+1],wires=i)
                    qml.RY(-pi/2, wires=i)
                
                et = torch.abs(param[-2]/param[-1])
                qml.ApproxTimeEvolution(H, et, 1)
                #dp = dephase_factor(et/self.t2)
                for i in range(self.num_qubit):
                    #qml.PhaseDamping(dp, wires=i)
                    qml.RY(pi/2, wires=i)
                
                # Phase accumulation
                qml.ApproxTimeEvolution(H_1, t, 1)
                for i in range(self.num_qubit):
                    qml.PhaseDamping(tau, wires=i)
                for i in range(self.num_qubit):
                    qml.RZ(param[i+self.num_qubit+1], wires=i)
                
                for i in range(self.num_qubit):
                    qml.RX(pi/2, wires=i)

            else:
                for i in range(self.num_qubit):
                    qml.RY(pi/2, wires=i)

                # Entangler
                et = torch.abs(param[-5]/param[-1])
                qml.ApproxTimeEvolution(H, et, 1)
                #dp = dephase_factor(et/self.t2)
                for i in range(self.num_qubit):
                    #qml.PhaseDamping(dp, wires=i)
                    qml.RX(param[i+1],wires=i)
                    qml.RY(-pi/2, wires=i)
                
                et = torch.abs(param[-4]/param[-1])
                qml.ApproxTimeEvolution(self.H, et, 1)
                #dp = dephase_factor(et/self.t2)
                for i in range(self.num_qubit):
                    #qml.PhaseDamping(dp, wires=i)
                    qml.RY(pi/2, wires=i)

                # Entangler
                et = torch.abs(param[-3]/param[-1])
                qml.ApproxTimeEvolution(H, et, 1)
                #dp = dephase_factor(et/self.t2)
                for i in range(self.num_qubit):
                    #qml.PhaseDamping(dp, wires=i)
                    qml.RX(param[i+5],wires=i)
                    qml.RY(-pi/2, wires=i)
                
                et = torch.abs(param[-2]/param[-1])
                qml.ApproxTimeEvolution(H, et, 1)
                #dp = dephase_factor(et/self.t2)
                for i in range(self.num_qubit):
                    #qml.PhaseDamping(dp, wires=i)
                    qml.RY(pi/2, wires=i)
                
                # Phase accumulation
                qml.ApproxTimeEvolution(H_1, t, 1)
                for i in range(self.num_qubit):
                    qml.PhaseDamping(tau, wires=i)
                for i in range(self.num_qubit):
                    qml.RZ(param[i+9], wires=i)
                
                for i in range(self.num_qubit):
                    qml.RX(pi/2, wires=i)

            return qml.density_matrix(wires = range(self.num_qubit))

        # fig, ax = qml.draw_mpl(circuit)(torch.cat((torch.tensor([0.0]),self.params)))
        # plt.show()

        if not self.no_ps:

            self.inner_circuit = circuit

            @qml.qnode(self.dev, interface = 'torch', diff_method = 'backprop')
            def post_selection(param: torch.Tensor):
            
                rho = self.inner_circuit(param)
                numerator = self.K @ rho @ self.K.conj().T
                denominator = torch.trace(numerator)

                rho_ps = numerator / denominator

                qml.QubitDensityMatrix(rho_ps, wires=range(self.num_qubit))
                return qml.density_matrix(wires = range(self.num_qubit))

            self.circuit = post_selection
        else:
            self.circuit = circuit

    def set_param(self):
        '''
        Reset parameters.

        Args:

        None

        Returns:

        None. Instead, reset all parameters to pi/2.
        '''
        template = [pi/2, pi/2]
        #size = (num_params[self.num_qubit-1],)

        #template = pi*(np.random.random(size))/self.num_qubit

        self.params = self.params = torch.tensor(template, requires_grad=False)

    def fmod_param(self):
        
        template = self.params.clone().detach().numpy()
        template = np.fmod(template,2*pi)
        self.params = torch.tensor(template, requires_grad = True)
