# %%
from copy import deepcopy
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy import interpolate, integrate
from scipy.signal import windows

from qutip import qeye, tensor, destroy, basis
import qutip
import qutip_qip
from qutip_qip.device import Model, SCQubitsModel, ModelProcessor
from qutip_qip.operations import expand_operator, Gate, gate_sequence_product
from qutip_qip.compiler import GateCompiler, Instruction
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
            "g": [0.01] * self.num_qubits,
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

            """
            #Add off-diagonal coupling terms
            if m != self.num_qubits - 1:
                a1 = tensor([destroy(self.dims[m]), qeye(self.dims[m + 1])])
                a2 = tensor([qeye(self.dims[m]), destroy(self.dims[m + 1])])
                self._drift.append(
                    (
                        2
                        * np.pi
                        * self.params["g"][m]
                        * (a1 * a2.dag() + a1.dag() * a2),
                        [m, m + 1],
                    )
                )
            """
            
            
            

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
        for m in range(self.num_qubits):
            m1 = num_qubits - m - 1
            if m != self.num_qubits - 1:
                a1 = tensor([destroy(self.dims[m1]), qeye(self.dims[m1 - 1])])
                a2 = tensor([qeye(self.dims[m1]), destroy(self.dims[m1 - 1])])
                controls["Y" + str(m1) + str(m1 - 1)] = (
                    2 * np.pi * (a1 * a2.dag() + a1.dag() * a2),
                    [m1, m1 - 1]
                    )
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
        controls = gate.controls
        targets = gate.targets
        wq_L = max(self.params["wq"][controls[0]], self.params["wq"][targets[0]])
        wq_S = min(self.params["wq"][controls[0]], self.params["wq"][targets[0]])
        if (wq_L == self.params["wq"][controls[0]]):
            q_L = controls[0]
        else:
            q_L = targets[0]
        # Bring 20 and 11 closer
        detuning = (
            wq_L - wq_S
            + self.params["alpha"][targets[0]]
        )
        t_tol = gate.arg_value["t_tol"]  # Total time
        t_r = gate.arg_value["t_r"]  # Rising time
        sample_N = 2000
        tlist = np.linspace(0, t_tol, sample_N + 1)
        interval = t_tol / (sample_N)
        tlist_rz = np.linspace(interval + t_tol, interval + 2 * t_tol, sample_N + 1)
        tlist_all = np.concatenate([tlist, tlist_rz], axis = 0)
        max_coup = self.params["g"][controls[0]]
        max_amp = -gate.arg_value["strength_ratio"] * detuning  # Tuning strength
        #Define the Slepian pulse for the frequency
        def _optimal_Slepian_pulse(t, t_tol):
            t0 = t_tol/2
            coeff_g = 2 * math.sqrt(2) * max_coup
            thetai = math.atan(coeff_g / detuning)
            thetaf = math.atan(coeff_g / (detuning + max_amp))
            x = np.linspace(0, 1, sample_N + 1)
            y = windows.dpss(sample_N + 1, 3)
            f = interpolate.interp1d(x, y)
            fun1 = lambda t, tp: (integrate.quad(f, 0, t /(tp + 1e-3))[0]/integrate.quad(f, 0, 1)[0]) * (thetaf - thetai) + thetai
            fun2 = lambda t, tp: -(integrate.quad(f, 0, (t -tp)/(tp + 1e-3))[0]/integrate.quad(f, 0, 1)[0]) * (thetaf - thetai) + thetaf
            pulse = np.zeros(t.shape[0])
            for ind, td in enumerate(t):
                if(0 <= td < t0 ): 
                    pulse[ind] = coeff_g / math.tan(fun1(td, t0)) - detuning
                elif( td > t0):
                    pulse[ind] = coeff_g / math.tan(fun2(td, t0)) - detuning
                else:
                    pulse[ind] = coeff_g / math.tan(thetaf) - detuning
            return pulse
        
        # With rising time of coupling strength (currently we are not using it)
        #coeff = max_amp * _normalized_Hann_square_pulse(tlist, t_r, t_tol)
        #coeff_coup = max_coup * _normalized_Inv_Gaussian_pulse(tlist, t_r_coup, t_tol)

        coeff_zero = np.zeros(sample_N + 1)
        #pulse shape being Slepian
        #coeff_sz = _optimal_Slepian_pulse(tlist, t_tol)
        #Without rising time & pulse shape being Hann
        coeff_coup = np.full(shape = [sample_N + 1,], fill_value = max_coup)
        coeff_sz = max_amp * _normalized_Hann_square_pulse(tlist, t_r, t_tol)
        coeff1 = np.concatenate([coeff_sz, -coeff_sz], axis = 0)
        coeff_coup = np.concatenate([coeff_coup, coeff_zero], axis = 0)
        pulse_info = [
            ("sz" + str(q_L), coeff1),
            ("Y" + str(controls[0]) + str(targets[0]), coeff_coup),
        ]
        return [Instruction(gate, tlist_all, pulse_info)]

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


# %%
# Single-qubit model test
print("Single-qubit model:")
## Single qubit model X gate performance
num_qubits = 1
circuit = QubitCircuit(num_qubits)
gates = [Gate("SNOT", targets=[0])]
for gate in gates:
    circuit.add_gate(gate)
device = SCQubits2(num_qubits)
device.load_circuit(circuit)

H0 = device.get_qobjevo(noisy=True)[0](0)
eigenvalues = np.diag(H0.full())
phase_frame_U = get_phase_frame(
    device.get_full_tlist()[-1], eigenvalues, H0.dims
)

quobjevo, c_ops = device.get_qobjevo(noisy=True)
U_list = qutip.propagator(
    quobjevo.to_list(), c_op_list=c_ops, t=quobjevo.tlist
)
print(quobjevo.tlist[-1])
numeric_gate = qutip.Qobj((phase_frame_U * U_list[-1])[:2, :2])
print(
    "X gate fidelity",
    qutip.average_gate_fidelity(numeric_gate, qutip.sigmax()),
)


## Single qubit model Y gate performance
num_qubits = 1
circuit = QubitCircuit(num_qubits)
gates = [Gate("Y", targets=[0])]
for gate in gates:
    circuit.add_gate(gate)
device = SCQubits2(num_qubits, pulse_mode = "discrete")
device.load_circuit(circuit)

H0 = device.get_qobjevo(noisy=True)[0](0)
eigenvalues = np.diag(H0.full())
phase_frame_U = get_phase_frame(
    device.get_full_tlist()[-1], eigenvalues, H0.dims
)

quobjevo, c_ops = device.get_qobjevo(noisy=True)
U_list = qutip.propagator(
    quobjevo.to_list(), c_op_list=c_ops, t=quobjevo.tlist
)
print(quobjevo.tlist[-1])
numeric_gate = qutip.Qobj((phase_frame_U * U_list[-1])[:2, :2])
print(
    "Y gate fidelity",
    qutip.average_gate_fidelity(numeric_gate, qutip.sigmay()),
)

# %%############################
print()
# Two qubit model test
print("Two-qubit model without coupling")
## No coupling
### Single qubit model single-qubit gate sequence performance
num_qubits = 2
circuit = QubitCircuit(num_qubits)
gates = [
    Gate("X", targets=[0]),
    Gate("Y", targets=[1]),
    Gate("Y", targets=[0]),
    Gate("X", targets=[0]),
]
for gate in gates:
    circuit.add_gate(gate)
# Set coupling to 0, test single-qubit gates
device = SCQubits2(num_qubits, g=[0.0])
device.load_circuit(circuit)

H0 = device.get_qobjevo(noisy=True)[0](0)
eigenvalues = np.diag(H0.full())
phase_frame_U = get_phase_frame(
    device.get_full_tlist()[-1], eigenvalues, H0.dims
)
quobjevo, c_ops = device.get_qobjevo(noisy=True)
U_list = qutip.propagator(
    quobjevo.to_list(), c_op_list=c_ops, t=quobjevo.tlist
)

numeric_gate = qutip.Qobj(
    get_subspace_array(phase_frame_U * U_list[-1], np.array([0, 1, 3, 4])),
    dims=[[2, 2], [2, 2]],
)

ideal_unitary = circuit.compute_unitary()
print(
    "Single-qubit gate sequence fidelity",
    qutip.average_gate_fidelity(numeric_gate, ideal_unitary),
)

#%%##############
## With coupling
print("Two-qubit model with coupling")
# Ideally we should drive with the dressed frequency, but here we still use the bare frequency.
# Require dressed frame transformation!!!!!!!!!!!!!!!!!!!
### Single qubit model single-qubit gate sequence performance
num_qubits = 2
circuit = QubitCircuit(num_qubits)
gates = [ 
    Gate("X", targets=[0]),
    Gate("Y", targets=[1]),
    Gate("Y", targets=[0]),
    Gate("X", targets=[1]),
    # Gate("Z", targets=[1]),
]
for gate in gates:
    circuit.add_gate(gate)
# Set coupling to 0, test only single-qubit gates
device = SCQubits2(num_qubits)
device.load_circuit(circuit)

H0 = device.get_qobjevo(noisy=True)[0](0)
eigenvalues, dressed_frame_U = compute_dressed_frame(H0)

bare_eigenvalues = np.diag(np.real(H0))
phase_frame_U = get_phase_frame(
    device.get_full_tlist()[-1], eigenvalues, H0.dims
)
quobjevo, c_ops = device.get_qobjevo(noisy=True)
U_list = qutip.propagator(
    quobjevo.to_list(), c_op_list=c_ops, t=quobjevo.tlist
)
print(quobjevo.tlist[-1])
# Dressed frame transformation
numeric_gate = (
    phase_frame_U * dressed_frame_U.dag() * U_list[-1] * dressed_frame_U
)
numeric_gate = qutip.Qobj(
    get_subspace_array(numeric_gate, np.array([0, 1, 3, 4])),
    dims=[[2, 2], [2, 2]],
)

ideal_unitary = circuit.compute_unitary()
print(
    "Single-qubit gate sequence fidelity",
    qutip.average_gate_fidelity(numeric_gate, ideal_unitary),
)


#%%
#Two qubit gate with direct coupling
num_qubits = 2
circuit = QubitCircuit(num_qubits)
# CZ gate drive parameters
params = {"t_tol": 81, "t_r": 10.0, "strength_ratio": 0.9}
gates = {
        Gate("CSIGN", controls = [0], targets=[1], arg_value = params),
}
for gate in gates:
    circuit.add_gate(gate)

device = SCQubits2(num_qubits)
device.load_circuit(circuit)
device.plot_pulses(pulse_labels = [{"sz0":"sz0"},{"Y01":"Y01"}])
# Unitary evolution
quobjevo, c_ops = device.get_qobjevo(noisy=True)
U_list = qutip.propagator(
    quobjevo.to_list(), c_op_list=c_ops, t=quobjevo.tlist
)
# Since only final result matters, we could add phase_frame_U only in the end. 
# Here is a simplified version.
U = U_list[-1]
H = device.get_qobjevo(noisy=True)[0](0)
eigenvalues, dressed_frame_U = compute_dressed_frame(H)
phase_frame_U = get_phase_frame(
    device.get_full_tlist()[-1], eigenvalues, H.dims
)
numeric_gate_qubits = qutip.Qobj(
    get_subspace_array(phase_frame_U * U, np.array([0, 1, 3, 4])), 
    dims=[[2, 2], [2, 2]]
)
ZZ_phase = np.angle(
        numeric_gate_qubits[3, 3]
        / numeric_gate_qubits[2, 2]
        / numeric_gate_qubits[1, 1]
        * numeric_gate_qubits[0, 0]
    )
fid = qutip.average_gate_fidelity(
        qutip.Qobj(numeric_gate_qubits, dims=[[2, 2], [2, 2]]),
        qutip.Qobj(np.diag([1, 1, 1, -1]), dims=[[2, 2], [2, 2]]),
    )   
print(ZZ_phase)
print(fid)



#%%
#Here is the original version. Since here we view CZ as CZ + Rz to correct the fid,
#Two different phase_frame_U are used. 
ZZ_phase_result = []
fid_result = []
relative_ZZ_phase_result = []
H0 = device.get_qobjevo(noisy=True)[0](0.1)
for j, U in enumerate(U_list):
    eigenvalues, dressed_frame_U = compute_dressed_frame(H0)
    phase_frame_U = get_phase_frame(
        device.get_full_tlist()[j], eigenvalues, H0.dims
    )
    # Here, since we add g as a control pulse, we no longer need dressed frame correction
    ## However, rotating frame correction maybe needed.
    numeric_gate_qubits = qutip.Qobj(
        get_subspace_array(phase_frame_U * U, np.array([0, 1, 3, 4])),
        dims=[[2, 2], [2, 2]],
    )
    
    ## ZI phase caused by the sz0 drive
    ZI_phase_correction = (
        -1.0j
        * np.angle(numeric_gate_qubits[2, 2] / numeric_gate_qubits[0, 0])
        * tensor([qutip.num(2), qeye(2)])
    ).expm()
    numeric_gate_qubits = ZI_phase_correction * numeric_gate_qubits
    # Compute ZZ phase
    ZZ_phase = np.angle(
        numeric_gate_qubits[3, 3]
        / numeric_gate_qubits[2, 2]
        / numeric_gate_qubits[1, 1]
        * numeric_gate_qubits[0, 0]
    )
    ZZ_phase_result.append(ZZ_phase)
    relative_ZZ_phase = min(abs(ZZ_phase - np.pi), abs(ZZ_phase + np.pi))
    relative_ZZ_phase_result.append(relative_ZZ_phase)
    # Compute fidelity
    fid = qutip.average_gate_fidelity(
        qutip.Qobj(numeric_gate_qubits, dims=[[2, 2], [2, 2]]),
        qutip.Qobj(np.diag([1, 1, 1, -1]), dims=[[2, 2], [2, 2]]),
    )
    fid_result.append(fid)

plt.plot(device.get_full_tlist(), ZZ_phase_result)
plt.xlabel("Gate time [ns]")
plt.ylabel("ZZ phase accumulated")
plt.savefig('acc_ZZ_phase_56.2.pdf')
plt.show()
plt.plot(device.get_full_tlist(), fid_result)
plt.xlabel("Gate time [ns]")
plt.ylabel("CZ fidelity")
plt.show()
print("Accumulated ZZ phase", ZZ_phase_result[-1])
print("Accumulated ZZ phase at end of CZ", ZZ_phase_result[2001])
print("CZ fidelity", fid_result[-1])
opt_gate_time_index = np.argmax(fid_result)
optZZphase_gate_time_index = np.argmin(relative_ZZ_phase_result)
print("Optimal CZ fidelity", fid_result[opt_gate_time_index])

#  %%
#  Small circuit simulation 
#  Case 1: 3-Qubit Deutsch-Jozsa Algorithm 
qc0 = QubitCircuit(3, reverse_states = False)
params = {"t_tol": 56.2, "t_r": 10.0, "strength_ratio": 0.9}
Gates = {
    Gate("X", targets = 2),
    Gate("SNOT", targets = 0),
    Gate("SNOT", targets = 1),
    Gate("CSIGN", controls = 0, targets = 2,arg_value = params),
    Gate("CSIGN", controls = 1, targets = 2,arg_value = params),
    Gate("SNOT", targets = 0),
    Gate("SNOT", targets = 1),
    Gate("SNOT", targets = 2),
}

for gate in Gates:
    qc0.add_gate(gate)

def run_circuit(circuit):
    device = SCQubits2(num_qubits)
    device.load_circuit(circuit)

    # Dressed frame
    H0 = device.get_qobjevo(noisy=True)[0](0)
    eigenvalues, dressed_frame_U = compute_dressed_frame(H0)

    # Unitary evolution
    quobjevo, c_ops = device.get_qobjevo(noisy=True)
    U_list = qutip.propagator(
        quobjevo.to_list(), c_op_list=c_ops, t=quobjevo.tlist
    )

    # Eigenenergies
    eigenvalues_list = []
    for t in device.get_full_tlist():
        H = quobjevo(t)
        eigenvalues_t, _ = H.eigenstates()
        eigenvalues_list.append(eigenvalues_t)


    ZZ_phase_result = []
    fid_result = []
    relative_ZZ_phase_result = []
    for j, U in enumerate(U_list):
        phase_frame_U = get_phase_frame(
            device.get_full_tlist()[j], eigenvalues, H0.dims
        )
        # Rotating frame correction + dressed frame correction
        ## There is not phase_frame_U.dag()!! Unlike the dress frame transformation, the rotating frame transformation is a time-dependent one. At at t=0, it is identity.
        numeric_gate = phase_frame_U * dressed_frame_U.dag() * U * dressed_frame_U
        numeric_gate_qubits = qutip.Qobj(
            get_subspace_array(numeric_gate, np.array([0, 1, 3, 4])),
            dims=[[2, 2], [2, 2]],
        )
        ## ZI phase caused by the sz0 drive
        ZI_phase_correction = (
            -1.0j
            * np.angle(numeric_gate_qubits[2, 2] / numeric_gate_qubits[0, 0])
            * tensor([qutip.num(2), qeye(2)])
        ).expm()
        numeric_gate_qubits = ZI_phase_correction * numeric_gate_qubits
        # Compute ZZ phase
        ZZ_phase = np.angle(
            numeric_gate_qubits[3, 3]
            / numeric_gate_qubits[2, 2]
            / numeric_gate_qubits[1, 1]
            * numeric_gate_qubits[0, 0]
        )
        ZZ_phase_result.append(ZZ_phase)
        relative_ZZ_phase = min(abs(ZZ_phase - np.pi), abs(ZZ_phase + np.pi))
        relative_ZZ_phase_result.append(relative_ZZ_phase)

        ideal_unitary = circuit.compute_unitary()
        # Compute fidelity
        fid = qutip.average_gate_fidelity(
            qutip.Qobj(numeric_gate_qubits, dims=[[2, 2], [2, 2]]),
            ideal_unitary,
        )
        fid_result.append(fid)
        return ZZ_phase_result[-1], fid_result[-1]