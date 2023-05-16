# %%
from copy import deepcopy
from itertools import product
import numpy as np
import inspect
import random
import math
import time
import matplotlib.pyplot as plt

from scipy.special import erf
from scipy import interpolate, integrate
from scipy.signal import windows
from scipy.stats import chi2

from qutip import qeye, tensor, destroy, basis, rand_ket, Qobj
import qutip
import qutip_qip
from qutip_qip.device import Model, SCQubitsModel, ModelProcessor
from qutip_qip.operations import expand_operator, Gate, gate_sequence_product,x_gate, y_gate
from qutip_qip.compiler import GateCompiler, Instruction
from qutip_qip.algorithms import qft_gate_sequence
from qutip_qip.circuit import QubitCircuit
from qutip_qip.transpiler import to_chain_structure
np.set_printoptions(precision=5, linewidth=150, suppress=True)


class SCQubits2(ModelProcessor):
    def __init__(self, num_qubits, dims=None, zz_crosstalk=False, **params):
        if dims is None:
            dims = [3] * num_qubits
        model = SCQubitsModel2(
            num_qubits=num_qubits,
            dims=dims,
            zz_crosstalk=zz_crosstalk,
            **params,
        )
        super(SCQubits2, self).__init__(model=model)
        self.native_gates = ["RX", "RY", "CSIGN"]
        self._default_compiler = SCQubitsCompiler2
        self.pulse_mode = "continuous"

    def topology_map(self, qc):
        qc1 = to_chain_structure(qc, setup = "linear")
        return qc1
    
    def transpile(self, qc):
        qc = self.topology_map(qc)
        qc = qc.resolve_gates(basis=["RX", "RY", "CSIGN"])
        return qc


# Inherit from SCQubitsModel to reuse some class methods.
class SCQubitsModel2(SCQubitsModel):
    def __init__(self, num_qubits, dims=None, zz_crosstalk=False, **params):
        self.num_qubits = num_qubits
        self.dims = dims if dims is not None else [3] * num_qubits
        self.params = {
            # Instead of a rotating frame with respect to
            # each qubit's frequency,
            # here we only use a rotating frame fixed by one frequency.
            # The following wq will be added to the model by wq*a.dag()*a.
            # But one should still use the detuning related to
            # a reference frequency to improve the efficiency.
            # E.g. the number below could be two qubits of 5.5 GHz and 5 GHz.
            "wq": np.array(
                ((0.5, 0.0) * int(np.ceil(self.num_qubits / 2)))[
                    : self.num_qubits
                ]
            ),
            "alpha": [-0.3] * self.num_qubits,
            "g": [0.015] * self.num_qubits,
            "omega_single": [0.03] * self.num_qubits,
        }
        self.params.update(deepcopy(params))
        self._compute_params()
        self._drift = []
        self._set_up_drift()
        self._controls = self._set_up_controls()
        self._noise = []

    def _set_up_drift(self):
        for m in range(self.num_qubits):
            destroy_op = destroy(self.dims[m])
            coeff = 2 * np.pi * self.params["alpha"][m] / 2.0
            #Add anharmonicity 
            self._drift.append(
                (coeff * destroy_op.dag() ** 2 * destroy_op**2, [m])
            )
            # here important !!!!!!!!!!!!!!!!!!!!
            # Add constant frequency (qubit detuning)
            # Use one qubit as the reference frame
            self._drift.append(
                (
                    2
                    * np.pi
                    * self.params["wq"][m]
                    * destroy_op.dag()
                    * destroy_op,
                    [m],
                )
            )

        
    def _compute_params(self):
        """
        Compute the dressed frequency and the interaction strength.
        """
        pass

    
    def _set_up_controls(self):
       
        num_qubits = self.num_qubits
        dims = self.dims
        controls = super()._set_up_controls()
        for m in range(self.num_qubits):
            if m != self.num_qubits - 1:
                a1 = tensor([destroy(self.dims[m]), qeye(self.dims[m + 1])])
                a2 = tensor([qeye(self.dims[m]), destroy(self.dims[m + 1])])
                controls["Y" + str(m) + str(m + 1)] = (
                    2 * np.pi * (a1 * a2.dag() + a1.dag() * a2),
                    [m, m + 1]
                    )
        """
          for m in range(self.num_qubits):
            m1 = num_qubits - m - 1
            if m != self.num_qubits - 1:
                a1 = tensor([destroy(self.dims[m1]), qeye(self.dims[m1 - 1])])
                a2 = tensor([qeye(self.dims[m1]), destroy(self.dims[m1 - 1])])
                controls["Y" + str(m1) + str(m1 - 1)] = (
                    2 * np.pi * (a1 * a2.dag() + a1.dag() * a2),
                    [m1, m1 - 1]
                    )
        """
        return controls         
#A normalized pulse designed for modulating coupling
#with t_r_coup being the rising time of coupling
def _normalized_pulse_coup(t, t_r, t_tol):
    fun = lambda t, t_r, t_tol : np.sin(np.pi / 2 / t_r * t) ** 2
    t_r_coup = t_r / 20
    t_hold = t_tol - 2 * t_r
    maxvalue = fun(t_r, t_r, t_tol)
    pulse = maxvalue
    pulse = np.where(t < t_r, fun(t - t_r_coup, t_r, t_tol), pulse)
    pulse = np.where(t > t_tol - t_r, fun(t_tol - t_r_coup - t, t_r, t_tol), pulse)
    pulse = np.where(t < t_r_coup, 0, pulse)
    pulse = np.where(t > t_tol - t_r_coup, 0, pulse)
    return pulse

#A normalized Hann square pulse for frequency modulation
def _normalized_Hann_square_pulse(t, t_r, t_tol):
    fun = lambda t, t_r, t_tol: np.sin(np.pi / 2 / t_r * t) ** 2
    t_hold = t_tol - 2 * t_r
    max_value = fun(t_r, t_r, t_tol)
    pulse = max_value
    pulse = np.where(t < t_r, fun(t, t_r, t_tol), pulse)
    pulse = np.where(t > t_tol - t_r, fun(t_tol - t, t_r, t_tol), pulse)
    return pulse

# A Inverse Gaussian pulse
def _normalized_Inv_Gaussian_pulse(t, t_r, t_tol):
    sigma = t_r/4
    fun = lambda t, t_tol: (erf(t/sigma - 2) - erf((t - t_tol)/sigma+2))/2
    final_value = fun(t_tol, t_tol)
    pulse = np.where(t < t_tol, fun(t,t_tol),final_value)
    return pulse

# An Optimal Slepian pulse

# Approximations of Slepian using Hann pulse
def _compute_zi_phase_accumulation(tlist, coeff):
    return np.trapz(coeff, tlist)


class SCQubitsCompiler2(GateCompiler):
    def __init__(self, num_qubits, params):
        super(SCQubitsCompiler2, self).__init__(num_qubits, params=params)
        self.gate_compiler.update(
            {
                "RY": self.ry_compiler,
                "RX": self.rx_compiler,
                "CSIGN": self.cz_compiler,
            }
        )
        self.args = {  # Default configuration
            "shape": "hann",
            "num_samples": 501,
            "params": self.params,
            "DRAG": True,
        }

    def _rotation_compiler(self, gate, op_label, param_label, args):
        targets = gate.targets
        coeff, tlist = self.generate_pulse_shape(
            args["shape"],
            args["num_samples"],
            maximum=self.params[param_label][targets[0]],
            area=gate.arg_value / 2.0 / np.pi,
        )
        f = 2 * np.pi * self.params["wq"][targets[0]]
        if args["DRAG"]:
            pulse_info = self._drag_pulse(op_label, coeff, tlist, targets[0])
        elif op_label == "sx":
            pulse_info = [
                ("sx" + str(targets[0]), coeff),
                # Add zero here just to make it easier to add the driving frequency later.
                ("sy" + str(targets[0]), np.zeros(len(coeff))),
            ]
        elif op_label == "sy":
            pulse_info = [
                ("sx" + str(targets[0]), np.zeros(len(coeff))),
                ("sy" + str(targets[0]), coeff),
            ]
        else:
            raise RuntimeError("Unknown label.")
        return [Instruction(gate, tlist, pulse_info)]

    def cz_compiler(self, gate, args):
        controls = gate.controls[0]
        targets = gate.targets[0]
        wq_L = max(self.params["wq"][controls], self.params["wq"][targets])
        wq_S = min(self.params["wq"][controls], self.params["wq"][targets])
        fault_type = gate.arg_value["fault"][0]
        fault_arg = gate.arg_value["fault"][1]
        if (wq_L == self.params["wq"][controls]):
            q_L = controls
        else:
            q_L = targets
        # Bring 20 and 11 closer
        detuning = (
            wq_L - wq_S
            + self.params["alpha"][targets]
        )
        if(controls > targets):
            controls, targets = targets, controls
        t_tol = gate.arg_value["t_tol"]  # Total time
        t_r = gate.arg_value["t_r"]  # Rising time
        sample_N = 2000
        tlist = np.linspace(0, t_tol, sample_N + 1)
        t_r_coup = t_r / 5
        max_coup = self.params["g"][controls]
        max_amp = -gate.arg_value["strength_ratio"] * detuning  # Tuning strength
        # With rising time of coupling strength
        #coeff = max_amp * _normalized_Hann_square_pulse(tlist, t_r, t_tol)
        #coeff_coup = max_coup * _normalized_Hann_square_pulse(tlist, t_r_coup, t_tol)
        #Without rising time
        coeff_coup = np.full(shape = [sample_N + 1,], fill_value = max_coup)
        coeff_sz = max_amp * _normalized_Hann_square_pulse(tlist, t_r, t_tol)
        #Define the Slepian pulse for the frequency
        def _optimal_Slepian_pulse(t, t_tol, max_coup, detuning, max_amp):
            t0 = t_tol/2
            coeff_g = 2 * math.sqrt(2) * max_coup
            thetai = math.atan(coeff_g / detuning)
            thetaf = math.atan(coeff_g / (detuning + max_amp))
            x = np.linspace(0, 1, sample_N + 1)
            y = windows.dpss(sample_N + 1, 3)
            f = interpolate.interp1d(x, y)
            fun1 = lambda t, tp: (integrate.quad(f, 0, t /(tp + 1e-3), epsabs = 1e-10)[0]/integrate.quad(f, 0, 1, epsabs = 1e-10)[0]) * (thetaf - thetai) + thetai
            fun2 = lambda t, tp: -(integrate.quad(f, 0, (t -tp)/(tp + 1e-3), epsabs = 1e-10)[0]/integrate.quad(f, 0, 1, epsabs = 1e-10)[0]) * (thetaf - thetai) + thetaf
            pulse = np.zeros(t.shape[0])
            for ind, td in enumerate(t):
                if(0 <= td < t0 ): 
                    pulse[ind] = coeff_g / math.tan(fun1(td, t0)) - detuning
                elif(td > t0):
                    pulse[ind] = coeff_g / math.tan(fun2(td, t0)) - detuning
                else:
                    pulse[ind] = coeff_g / math.tan(thetaf) - detuning
            return pulse
        #coeff_sz = _optimal_Slepian_pulse(tlist, t_tol, max_coup, detuning, max_amp)


        #Add pulse-level control faults for CZ gate 
        if fault_type == 'missingZ':
            pulse_info = [
                ("Y" + str(controls) + str(targets), coeff_coup),
            ]

        elif fault_type == 'delay':
            interval = t_tol / sample_N
            add_counts = (int)(abs(fault_arg) * sample_N)
            t_fin = add_counts * interval
            tlist_add = np.linspace(t_tol + interval,t_tol + t_fin, add_counts)
            tlist_misalign = np.concatenate([tlist, tlist_add], axis = 0)
            tlist = tlist_misalign
            coeff_zero = np.zeros([add_counts,])
            if(fault_arg > 0):
                coeff_missz = np.concatenate([coeff_sz, coeff_zero], axis = 0)
                coeff_miscoup = np.concatenate([coeff_zero, coeff_coup], axis = 0)   
            else:
                coeff_missz = np.concatenate([coeff_zero, coeff_sz], axis = 0)
                coeff_miscoup = np.concatenate([coeff_coup, coeff_zero],axis = 0)
            pulse_info = [
                    ("sz" + str(q_L), coeff_missz),
                    ("Y" + str(controls) + str(targets), coeff_miscoup),
            ]
   
        elif fault_type == 'amp_sz':
            temp_amp = (1 - fault_arg) * max_amp
            coeff_sz_f = temp_amp * _normalized_Hann_square_pulse(tlist, t_r, t_tol)
            #coeff_sz_f = _optimal_Slepian_pulse(tlist, t_tol, max_coup, detuning, temp_amp)
            pulse_info = [
            ("sz" + str(q_L), coeff_sz_f),
            ("Y" + str(controls) + str(targets), coeff_coup),
        ]
        elif fault_type == 'amp_coup':
            temp_coup = (1 - fault_arg) * max_coup
            coeff_coup_f = np.full(shape = sample_N + 1, fill_value = temp_coup)
            #coeff_sz_f = _optimal_Slepian_pulse(tlist, t_tol, temp_coup, detuning, max_amp)
            pulse_info = [
            ("sz" + str(q_L), coeff_sz),
            ("Y" + str(controls) + str(targets), coeff_coup_f),
        ]
        elif fault_type == 'duration':
            t_tol_t = (1 + fault_arg) * t_tol
            tlist = np.linspace(0, t_tol_t, sample_N + 1)
            coeff_sz_t = max_amp * _normalized_Hann_square_pulse(tlist, t_r, t_tol_t)
            #coeff_sz_t = _optimal_Slepian_pulse(tlist, t_tol_t, max_coup, detuning, max_amp)
            pulse_info = [
            ("sz" + str(q_L), coeff_sz_t),
            ("Y" + str(controls) + str(targets), coeff_coup),
        ]
        else:
            pulse_info = [
            ("sz" + str(q_L), coeff_sz),
            ("Y" + str(controls) + str(targets), coeff_coup),
        ]
        return [Instruction(gate, tlist, pulse_info)]

    def _drag_pulse(self, op_label, coeff, tlist, target):
        dt_coeff = np.gradient(coeff, tlist[1] - tlist[0]) / 2 / np.pi
        # Y-DRAG
        alpha = self.params["alpha"][target]
        y_drag = -dt_coeff / alpha
        # Z-DRAG
        z_drag = -(coeff**2) / alpha + (np.sqrt(2) ** 2 * coeff**2) / (
            4 * alpha
        )
        # X-DRAG
        coeff += -(coeff**3 / (4 * alpha**2))

        pulse_info = [
            (op_label + str(target), coeff),
            ("sz" + str(target), z_drag),
        ]
        if op_label == "sx":
            pulse_info.append(("sy" + str(target), y_drag))
        elif op_label == "sy":
            pulse_info.append(("sx" + str(target), -y_drag))
        return pulse_info

    def ry_compiler(self, gate, args):
        return self._rotation_compiler(gate, "sy", "omega_single", args)

    def rx_compiler(self, gate, args):
        return self._rotation_compiler(gate, "sx", "omega_single", args)

    def compile(self, circuit, schedule_mode=None, args=None):
        """
        Add the oscillating phase of the drive that matches with the qubit frequency in
        the rotating wave approximation.
        Here we drive at the bare qubit frequency, but this could be improved.
        """
        compiled_tlist_map, compiled_coeffs_map = super().compile(
            circuit, schedule_mode=None, args=None
        )
        pulse_list = compiled_coeffs_map.keys()
        for q in range(self.num_qubits):
            if (
                "sx" + str(q) not in pulse_list
                and "sy" + str(q) not in pulse_list
            ):
                continue
            x_coeff = compiled_coeffs_map.get("sx" + str(q), 0.0)
            y_coeff = compiled_coeffs_map.get("sy" + str(q), 0.0)
            # Note that if neither tlist is None, they must be identical!
            tlist = None
            tlist_x = compiled_tlist_map.get("sx" + str(q), None)
            tlist_y = compiled_tlist_map.get("sy" + str(q), None)
            tlist = tlist_x if tlist_x is not None else tlist_y
            omega = x_coeff + 1.0j * y_coeff
            f = 2 * np.pi * self.params["wq"][q]
            omega *= np.exp(-1.0j * f * tlist)
            x_coeff = np.real(omega)
            y_coeff = np.imag(omega)
            compiled_coeffs_map["sx" + str(q)] = x_coeff
            compiled_coeffs_map["sy" + str(q)] = y_coeff
            compiled_tlist_map["sx" + str(q)] = tlist
            compiled_tlist_map["sy" + str(q)] = tlist
        return compiled_tlist_map, compiled_coeffs_map


def compute_dressed_frame(ham):
    """
    Compute the dressed state using the drift Hamiltonian.
    The eigenstates are saved as a transformation unitary, where
    each column corresponds to one eigenstate.
    The transformation matrix can be obtained by
    :obj:`.get_transformation`.
    """
    eigenvalues, U = np.linalg.eigh(ham.full())
    # Permute the eigenstates in U according to the overlap
    # with bare qubit states so that the transformed Hamiltonian
    # match the definition of logical qubits.
    # A logical qubit state |i> is defined as the eigenstate that
    # has the maximal overlap with the corresponding bare qubit state |i>.
    qubit_state_labels = np.argmax(np.abs(U), axis=0)
    if len(qubit_state_labels) != len(set(qubit_state_labels)):
        raise ValueError(
            "The definition of dressed states is ambiguous."
            "Please define the unitary manually by the attributes"
            "Processor.dressing_unitary"
        )
    eigenvalues = eigenvalues[np.argsort(qubit_state_labels)]
    U = U[:, np.argsort(qubit_state_labels)]  # Permutation
    sign = np.real(np.sign(np.diag(U)))
    U = U * sign
    U = qutip.Qobj(U, dims=ham.dims)
    return eigenvalues, U


def get_phase_frame(t, eigenvalues, dims):
    phase_frame_U = np.diag(np.exp((1.0j * eigenvalues * t)))
    return qutip.Qobj(phase_frame_U, dims=dims)


def get_subspace_array(qobj, indices):
    """Return the subspace unitary e.g. for 00,01,10,11,20,02"""
    return qobj.full()[
        indices[:, np.newaxis],
        indices,
    ]

#%%
### Here we provide a library for quantum circuits selected as test objects.
def inv_qft(N, swapping = True):
    qc = QubitCircuit(N)
    if(N == 1):
        qc.add_gate("SNOT", targets = [0])
    else:
        if swapping:
            for i in range(N // 2):
                qc.add_gate("SWAP", targets=[N - i - 1, i])
        for i in range(N - 1, -1, -1):
            qc.add_gate("SNOT", targets = [i])
            for j in range(i - 1, -1, -1):
                qc.add_gate(
                    "CPHASE",
                    targets = [j],
                    controls = [i],
                    arg_label = r"{-\pi/2^{%d}}" % (i - j),
                    arg_value = - np.pi / (2 ** (i - j))
                )
                
    return qc
#%%

### Simulation Benchmark: A 3-Qubit Deutsch Josza Algorithm
def DJ_alg(N = 3):
    Gates = {
        Gate("X", targets = [2]),
        Gate("SNOT", targets = [0]),
        Gate("SNOT", targets = [1]),
        Gate("SNOT", targets = [2]),
        Gate("CNOT", targets = [0]),
        Gate("CNOT", targets = [1]),
        Gate("SNOT", targets = [0]),
        Gate("SNOT", targets = [1]),
}
    qc = QubitCircuit(3)
    for gate in Gates:
        qc.add_gate(gate)
    return qc

### Simulation object I: A (N = 2n)-qubit Quantum Draper Adder (for test pattern selection and illustration of scalability)
def draper_adder(N):
    n = N // 2 # Number of classical bits
    qc = QubitCircuit(N)
    qc.add_circuit(qft_gate_sequence(n, swapping = False),start = n) #QFT for number a 
    for i in range(n):
       for j in range(n - i):
           qc.add_gate(
               "CPHASE",
               targets = [n + i],
               controls = [i + j],
               arg_label = r"{-\pi/2^{%d}}" % (j),
               arg_value = np.pi / (2**(j))
           ) 
    qc.add_circuit(inv_qft(n, swapping = False), start = n) # inv-QFT
    return qc
## Simulation object II: A N-qubit Quantum supremacy circuit
def random_circuit(N = 4):
    def user_gate1():
        exp4 = np.exp(1/4 * 1j * np.pi)
        mat = 1 / math.sqrt(2) * np.array([[exp4, - 1j],[1, exp4]])
        return Qobj(mat, dims = [[2], [2]])
    def user_gate2():
        ele = 1/2 * (1 + 1j) 
        mat = np.array([[ele, -ele],[ele, ele]])
        return Qobj(mat, dims = [[2], [2]])
    sq2 = 1/ math.sqrt(2)
    choice = [0, 1, 2]
    qc = QubitCircuit(N, user_gates = {"SQRTW": user_gate1, "SQRTY": user_gate2})
    gate_set = ["SQRTNOT", "SQRTY", "SQRTW"]
    control_set = [0, 0, 2, 1, 2, 1, 0]
    target_set = [1, 3, 3, 2, 3, 2, 3]
    for i in range(7):
        for j in range(N):
            p = (int)(np.random.choice(choice))
            qc.add_gate(gate_set[p],targets = [j])
        qc.add_gate("CSIGN", targets = [target_set[i]], controls = [control_set[i]])
        
    return qc


#%%
###Circuit Level Simulation
###First, we need to obtain the custom defined faulty CZ gate and faulty circuit generated by it.
def get_faulty_CZ(fault_type, fault_arg, ZI_pulse = False):
    """
    shape: pulse shapes for sz control. Support: 'Slepian', 'Hann'
    fault_type: pulse_level faults for the CZ gate. Support: 'missingZ', 'amp_sz', 'amp_coup', 'duration', 'delay'
    fault_arg: fault scale of selected faults. 
    ZI_pulse: require ZI phase correction or not. Default: False. If True, then ZI correction is added 
    and correction matrix is disabled. More experimental but cause fidelity loss. 
    """
    num_qubits = 2
    circuit = QubitCircuit(num_qubits)
    params = {"t_tol": 55.7, "t_r": 10.0, "strength_ratio": 0.9, 'fault':[fault_type, fault_arg]}
    circuit.add_gate(Gate("CSIGN", controls = [0], targets=[1], arg_value = params))
    device = SCQubits2(num_qubits)
    device.load_circuit(circuit)

    quobjevo, c_ops = device.get_qobjevo(noisy=True)
    U_list = qutip.propagator(quobjevo.to_list(), c_op_list=c_ops, t=quobjevo.tlist)
    U = U_list[-1]

    if(fault_type == 'delay'):
        index = -1 if fault_arg > 0 else 1
        H = device.get_qobjevo(noisy=True)[0](device.get_full_tlist()[index])
    else:
        H = device.get_qobjevo(noisy=True)[0](device.get_full_tlist()[-1])
    eigenvalues, dressed_frame_U = compute_dressed_frame(H)

    phase_frame_U = get_phase_frame(
        device.get_full_tlist()[-1], eigenvalues, H.dims
    )
    numeric_gate_qubits = qutip.Qobj(
        get_subspace_array(phase_frame_U * U, np.array([0, 1, 3, 4])), 
        dims=[[2, 2], [2, 2]]
    )  

    ZI_phase_correction = (
        -1.0j
        * np.angle(numeric_gate_qubits[2, 2] / numeric_gate_qubits[0, 0])
        * tensor([qutip.num(2), qeye(2)])
    ).expm()

    numeric_gate_qubits = ZI_phase_correction * numeric_gate_qubits
    return numeric_gate_qubits
### We try to introduce a small trick to help 
import cvxopt as cvx
import cvxpy as cp
def get_optim_state(fault_info):
    CZ = qutip.Qobj(np.diag([1, 1, 1, -1]), dims = [[2, 2], [2, 2]])
    fault_type, fault_arg = fault_info
    target_gate = CZ * get_faulty_CZ(fault_type, fault_arg)
    eigenvalues = np.array(qutip.Qobj.eigenenergies(target_gate))
    eigenvalues = np.flip(eigenvalues)
    print(eigenvalues)
    re = eigenvalues.real.copy()
    im = eigenvalues.imag.copy()
    ones = np.ones(4)
    zeros = np.zeros(4)
    x = cp.Variable(4)
    
    #constraints = [cp.sum(x) == 1, x>=zeros]
    #objective = cp.Minimize(cp.norm(eigenvalues @ x, 2))
    prob = cp.Problem(cp.Minimize(cp.square(re.T @ x)+cp.square(im.T @ x)),
                    [ones.T @ x == 1, 
                     x >= zeros])
    #prob = cp.Problem(objective, constraints)
    prob.solve()
    print(x.value)
    amp = np.sqrt(abs(x.value))
    optim_state = qutip.Qobj(np.array(amp), dims = [[2, 2],[1, 1]], shape = [4,1])
    return optim_state
print(get_optim_state(["duration", 0.05]))
#%%
## Then we replace the circuit's propagator with a faulty one, using faulty CZ gate to replace fault-free CZ gates. 
def faulty_propagator(num_qubits, qc, fault_info, expand = True):
    """
    In:
    num_qubits: dimension of the circuit.
    qc: 'QubitCircuit' object, gate information of circuits
    fault_info: 'list' object, specifies the location of fault, fault type and fault arg. 
    Out:
    U_list:
    overall_inds:
    Then one can use _gate_sequence_product/qc.run to perform the fault simulation.
    """
    U_list = []
    fault_index = 0
    gates = qc.gates
    cz_count = 0
    for index, gate in enumerate(gates):
        if gate.name == "GLOBALPHASE":
            qobj = gate.get_qobj(qc.N)
            U_list.append(qobj)
            continue

        if gate.name in qc.user_gates:
            if gate.controls is not None:
                raise ValueError(
                    "A user defined gate {} takes only  "
                    "`targets` variable.".format(gate.name)
                )
            func_or_oper = qc.user_gates[gate.name]
            if inspect.isfunction(func_or_oper):
                func = func_or_oper
                para_num = len(inspect.getfullargspec(func)[0])
                if para_num == 0:
                    qobj = func()
                elif para_num == 1:
                    qobj = func(gate.arg_value)
                else:
                    raise ValueError(
                        "gate function takes at most one parameters."
                    )
            elif isinstance(func_or_oper, Qobj):
                qobj = func_or_oper
            else:
                raise ValueError("gate is neither function nor operator")
            if expand:
                all_targets = gate.get_all_qubits()
                qobj = expand_operator(
                    qobj, dims=qc.dims, targets=all_targets
                )
        else:
            
            gate_count = 0
            if expand:
                if gate.name == 'CSIGN' and cz_count == 0:
                    fault_index = index
                    cz_count = cz_count + 1
                    [fault_type, fault_arg] = fault_info
                    qobj_compact = get_faulty_CZ(fault_type, fault_arg)
                    targets = gate.get_all_qubits()
                    qobj = expand_operator(qobj_compact, dims = qc.dims, targets = targets)
                else:
                    
                    qobj = gate.get_qobj(qc.N, qc.dims)
            else:
                if gate.name == 'CSIGN':
                    cz_count = cz_count + 1
                    #fault_index = index
                    [fault_type, fault_arg] = fault_info
                    targets = gate.get_all_qubits()
                    qobj = get_faulty_CZ(fault_type, fault_arg)
                else:
                    qobj = gate.get_compact_qobj()
        U_list.append(qobj)
    return U_list, fault_index, targets 

#%%
def back_propagation(num_qubits, qc, fault_info):
    """
    Reproduce the optimal test pattern for the  
    """
    N = num_qubits
    U_list, index, targets = faulty_propagator(num_qubits, qc, fault_info)
    state = np.array(get_optim_state(fault_info))
    state_exp = np.zeros(2**num_qubits, dtype = complex)
    s, l = targets
    if l < s:
        l, s = s, l
    for k in range(4):
        a = k // 2
        b = (k - a * 2)
        l1 = 2**(N - l - 1)
        s1 = 2**(N - s - 1)
        state_exp[a * s1 + b * l1] = state[k][0]
    state_exp = qutip.Qobj(state_exp, dims = [[2]*num_qubits, [1]* num_qubits], shape = [2**num_qubits, 1])
    qc = QubitCircuit(N)
    for j in range(N):
        if(j != s and j != l):
            qc.add_gate("SNOT", targets = [j])
    state_exp = qc.run(state_exp)
    U_list1 = U_list[0:index]
    for j in range(index - 1, -1, -1):
        U = U_list[j].dag()
        state_exp = U * state_exp 
    for j in range(N):
        state_amp = 
    return state_exp
qc = draper_adder(4).resolve_gates(basis = ["RX", "RY","CSIGN"])
print(back_propagation(4, qc, ['amp_coup',0.1]))

#%%%
### We also need some auxiliary functions for hypothesis testing

##Get the total number of CSIGN gates in the circuit, for fault specification
def get_CZ_count(qc):
    cz_count = 0
    for gate in qc.gates:
        if(gate.name == 'CSIGN'):
            cz_count = cz_count + 1
    return cz_count
## Get all n-qubit states. For a random state (random under Haar measure), directly use rand_ket
def traverse_states(num_qubits, res_info = [0, 0]):
    n = num_qubits
    N = 2**n
    index = np.zeros([N,n], dtype = int)
    states = []
    loc, val = res_info
    for i in range(N):
        temp = i
        for j in range(n):
            pow2 = 2**(n - j - 1)
            k = temp // pow2
            index[i,j] = k
            temp = temp - k * pow2
    for i in range(N):
        if(loc >= 1 and loc <= n and index[i,loc - 1] != val):
            continue
        else:
            for j in range(n):
                k = index[i,j]
                if(j == 0):
                    state = basis(2,k)
                else:
                    state = tensor(state, basis(2,k))
            states.append(state)                         
    return states
#%%
### Function of Hypothesis Testing. We don't calculate the fidelity anymore; 
### Instead, we really perform simulation and measurement to record the expected test repetitions to detect a single fault.
def hyp_testing(num_qubits, qc, fault_info, ini_state):
    """
    qc: ideal Quantum circuit, with CSIGN gate being a fault-free one
    fault_info: configuration of faults for CZ gate. 
    ini_state: initial state for testing. 
    """
    fin_state_t = qc.run(ini_state)
    U_list, fault_index, targets = faulty_propagator(num_qubits, qc, fault_info, expand = True)
    fin_state_f = qc.run(ini_state, U_list = U_list)
    pown = 2**num_qubits
    valid_index = [] # Record the valid (!=0) components of the fault-free final state
    fin_amp_t = np.zeros(pown) # Amplitude form of the ideal probability distribution
    fin_amp_f = np.zeros(pown) # Amplitude form of the ideal probability distribution
    for i in range(pown): 
        if(abs(fin_state_t[i]) != 0):
            valid_index.append(i)
        fin_amp_t[i] = abs(fin_state_t[i])**2
        fin_amp_f[i] = abs(fin_state_f[i])**2
    
    fin_amp_t = fin_amp_t/np.sum(fin_amp_t)
    fin_amp_f = fin_amp_f/np.sum(fin_amp_f)
    length = np.shape(valid_index)[0]
    choice = np.linspace(0, pown - 1, pown)
    N = 100
    count_avg = 0
    for k in range(N):
        freq_f = np.zeros(pown)
        freq_t = np.zeros(pown)
        quit_sig = 0
        count = 0
        while(quit_sig == 0 and count < 5e3): 
            count = count + 1
            freq_t = count * fin_amp_t
            sample = (int)(np.random.choice(choice, p = fin_amp_f))
            chi2_real = 0
            if(np.isin(sample, valid_index) == False):
                quit_sig = 1
            else:
                freq_f[sample] += 1
                for i, index in enumerate(valid_index):
                    chi2_real += (freq_f[index]  - freq_t[index])**2 / freq_t[index]
                if(chi2_real > chi2.ppf(0.95, length - 1)):
                    quit_sig = 1
        #print(count)
        count_avg += count
    count_avg = count_avg / N
    return count_avg
#%%
#methodI: Gate-level simulation
num_qubits = 4
qc = random_circuit(num_qubits)

#%%
Name = []
Targets = []
for gate in qc.gates:
    Name.append(gate.name)
    if(gate.controls != None):
        Targets.append([gate.targets[0],gate.controls[0]])
    else:
        Targets.append([gate.targets[0]])

file = open('Gatelist for supremacy.txt','w')
file.write(str([Name,Targets]))
file.close()
#%%
num_qubits = 4
qc_add = draper_adder(num_qubits)
qc_add = qc_add.resolve_gates(basis = ["RX","RY","CSIGN"])
states = traverse_states(num_qubits, [0, 0])
test_rep_duration = np.zeros(2**num_qubits, dtype = int)
test_rep_sz = np.zeros(2**num_qubits, dtype = int)
test_rep_coup = np.zeros(2**num_qubits, dtype = int)
test_rep_delay = np.zeros(2**num_qubits, dtype = int)
for index, state in enumerate(states):
    print(index)
    test_rep_duration[index] = math.ceil(hyp_testing(num_qubits, qc_add, ['duration',0.1], state))
    test_rep_sz[index] = math.ceil(hyp_testing(num_qubits, qc_add, ['amp_sz',0.1], state))
    test_rep_coup[index] = math.ceil(hyp_testing(num_qubits, qc_add, ['amp_coup',0.1], state))
    test_rep_delay[index] = math.ceil(hyp_testing(num_qubits, qc_add, ['delay',0.1], state))
#%%
print(test_rep_delay)
print(test_rep_sz)
print(test_rep_coup)
print(test_rep_duration)
#%%
np.savetxt('delay2-1.txt', test_rep_delay2, fmt = "%d")
np.savetxt('duration2-1.txt',test_rep_duration2, fmt = "%d")
np.savetxt('sz2-1.txt',test_rep_sz2, fmt = "%d")
np.savetxt('coup2-1.txt',test_rep_coup2, fmt = "%d")
# %%
### Evaluate and plot the results of the Quantum Draper Adder
def generate_binary_strings(n):
    return [format(i, f'0{n}b') for i in range(2**n)]
X = generate_binary_strings(num_qubits)

plt.figure(figsize = (8,6))
for i in range(16):
    plt.bar(X[i], test_rep_duration[i],color = 'cornflowerblue', width = 0.6)
    
for x,y in zip(X,test_rep_duration):
    plt.text(x,y,y, ha='center',va='bottom')

plt.title("test repetition of each pattern with 10% duration fault",fontweight = 'bold',fontsize = 18)

plt.xlabel("test pattern", fontweight = 'bold', fontsize = 14)

plt.ylabel("test repetition", fontweight = 'bold', fontsize = 14)

plt.savefig('pattern-repduration10.eps')
# %%
