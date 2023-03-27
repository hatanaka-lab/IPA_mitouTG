import sys, time, random, json
import numpy as np

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils.mitigation import CompleteMeasFitter, complete_meas_cal
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import CircuitSampler, StateFn, PrimitiveOp, ListOp, PauliExpectation, PauliSumOp
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.providers.aer import StatevectorSimulator
from qiskit_experiments.library import StateTomography
from qiskit_experiments.framework import ExperimentData
from qiskit import execute

from scipy.optimize import minimize, OptimizeResult



def calc_expvals(prm, *fargs):
    """ calculates the expectation value of given operators.
    Args:
        prm(np.array(float)): ansatz parameters
    fargs:
        0: sampler(CircuitSampler): sampler. if None, matrix calculation is conducted.
        1: ansatz(CuantumCircuit): ansatz circuit
        2: qubit_ops(list[list[tuple(str,complex)]]): list of pauli operators to evaluate
        3: disp(int): print level
        4: user_messenger(UserMessenger): publish the interim results
    Returns:
        result(list(float)): expectation value of the cost function
    """
    state_fn = StateFn(fargs[1].bind_parameters(prm))
    if fargs[0] is not None:
        # measure pauli terms individually
        pauli_tmp = [[op_[0] for op_ in op] for op in fargs[2]]
        all_paulis = list(set(sum(pauli_tmp, [])))
        all_pauliop = [PrimitiveOp(Pauli(op)) for op in all_paulis]
        pauli_index = [[all_paulis.index(op_) for op_ in op] for op in pauli_tmp]
        all_coeffs = [[op_[1] for op_ in op] for op in fargs[2]]
        # measure
        meas_circ = ~StateFn(ListOp(all_pauliop)) @ state_fn
        val = fargs[0].convert(PauliExpectation(group_paulis=True).convert(meas_circ)).eval()
        res_tmp = [[val[idx] for idx in p_idx] for p_idx in pauli_index]
        result = [np.sum(np.array(coeff) * np.array(res)).real for coeff,res in zip(all_coeffs, res_tmp)]
        if fargs[3] == 2:
            fargs[-1].publish(f"sampled: {val} for {all_paulis}")
            fargs[-1].publish(f"prm= {prm} val= {result}")
        elif fargs[3] == 1:
            fargs[-1].publish(f"prm= {prm} val= {result}")
    else:
        mat_ops = [SparsePauliOp.from_list(op).to_matrix() for op in fargs[2]]
        result = [np.dot(state_fn.adjoint().to_matrix(), np.dot(mop, state_fn.to_matrix())).real for mop in mat_ops]
        if fargs[3] == 2 or fargs[3] == 1:
            fargs[-1].publish(f"prm= {prm} val= {result}")
    return result



def calc_expvals_jac(prm, *fargs, previous_result=None):
    """ calculates the expectation value of given operators.
    Args:
        prm(np.array(float)): ansatz parameters
    fargs:
        0: jac_set(dict): settings for differentiation
        1: sampler(CircuitSampler): sampler. if None, matrix calculation is conducted.
        2: ansatz(CuantumCircuit): ansatz circuit
        3: qubit_ops(list[list[tuple(str,complex)]]): list of pauli operators to evaluate
        4: disp(int): print level
        5: user_messenger(UserMessenger): publish the interim results
    Returns:
        result(np.array(op_idx, prm_idx)): jacobian of the cost function
    """
    if abs(1.0 - fargs[0].get("sample_rate", 1.0)) > 1e-12:
        assert (fargs[0]["sample_rate"] < 1.0) and (fargs[0]["sample_rate"] > 0.0), f"sample_rate={fargs[0]['sample_rate']} is invalid."
        sample_no = int(len(prm) * fargs[0]["sample_rate"])
        sample_idx = sorted(random.sample(list(range(len(prm))), sample_no))
        if (fargs[0].get("padding", "zero") == "previous") and (previous_result is not None):
            jac = np.zeros((len(fargs[3]),len(prm)))
            jac = [[0.0 if p_idx in sample_idx else previous_result[op_idx,p_idx] for p_idx in range(len(prm))] for op_idx in range(len(fargs[3]))]
            jac = np.array(jac, dtype=float)
        else:
            jac = np.zeros((len(fargs[3]),len(prm)))
    else:
        sample_idx = range(len(prm))
        jac = np.zeros((len(fargs[3]),len(prm)))
    prm_tmp = prm.copy()
    if fargs[0].get("numeric", False):
        delta = fargs[0].get("delta", 0.1)
        for idx in sample_idx:
            prm_tmp[idx] += delta/2.0
            plus = calc_expvals(prm_tmp, *fargs[1:])
            prm_tmp[idx] -= delta
            minus = calc_expvals(prm_tmp, *fargs[1:])
            prm_tmp[idx] += delta/2.0
            jac[:,idx] = (np.array(plus) - np.array(minus)) / delta

    elif fargs[0].get("term", 2) == 2:
        shift = fargs[0].get("shift", np.pi/2.0)
        for idx in sample_idx:
            prm_tmp[idx] += shift
            plus = calc_expvals(prm_tmp, *fargs[1:])
            prm_tmp[idx] -= shift * 2.0
            minus = calc_expvals(prm_tmp, *fargs[1:])
            prm_tmp[idx] += shift
            jac[:,idx] = (np.array(plus) - np.array(minus)) / (2.0 * np.sin(shift))
    elif fargs[0].get("term", 2) == 4:
        shift = fargs[0].get("shift", [np.pi/2.0, np.pi])
        d = fargs[0].get("d", [0.5, (np.sqrt(2)-1.0)/4.0])
        assert len(shift) == 2, "the length of the 'shift' is invalid."
        assert len(d) == 2, "the length of the 'd' is invalid."
        for idx in sample_idx:
            prm_tmp[idx] += shift[0]
            plus_0 = calc_expvals(prm_tmp, *fargs[1:])
            prm_tmp[idx] -= shift[0] * 2.0 
            minus_0 = calc_expvals(prm_tmp, *fargs[1:])
            prm_tmp[idx] += shift[0] + shift[1]
            plus_1 = calc_expvals(prm_tmp, *fargs[1:])
            prm_tmp[idx] -= shift[1] * 2.0
            minus_1 = calc_expvals(prm_tmp, *fargs[1:])
            prm_tmp[idx] += shift[1]
            jac[:,idx] = d[0] * (np.array(plus_0) - np.array(minus_0))
            jac[:,idx] -= d[1] * (np.array(plus_1) - np.array(minus_1))
    return jac            



def calc_overlap(prm1, prm2, ansatz1, ansatz2, *fargs):
    """ calculates the expectation value of given operators.
    Args:
        prm(np.array(float)): ansatz parameters for ansatz1
        prm(np.array(float)): ansatz parameters for ansatz2
        ansatz1(CuantumCircuit): ansatz circuit 1
        ansatz2(CuantumCircuit): ansatz circuit 2
    fargs:
        0: sampler(CircuitSampler): sampler
        1: ansatz(CuantumCircuit): ansatz circuit
        2: qubit_ops(list[list[tuple(str,complex)]]): list of pauli operators (not used)
        3: disp(int): print level
        4: user_messenger(UserMessenger): publish the interim results
    Returns:
        result(complex): overlap value <psi_1|psi_2> (NOT squared)
    """
    if ansatz2 is None:
        ansatz2 = ansatz1
    state_fn1 = StateFn(ansatz1.bind_parameters(prm1))
    state_fn2 = StateFn(ansatz2.bind_parameters(prm2))
    # measure
    if fargs[0] is not None:
        meas_circ = ~state_fn1 @ state_fn2
        result = fargs[0].convert(PauliExpectation(group_paulis=True).convert(meas_circ)).eval()
    else:
        result = np.dot(state_fn1.adjoint().to_matrix(), state_fn2.to_matrix())
    return result



class Calculator:
    def __init__(self, operators, sampler, ansatz, user_messenger, init_prm, if_vqd, vqd_set, if_vqeac, vqeac_set, jac_set):
        self.eval_count = 0
        self.op_name = ["ene", "tot_n", "ssq", "sz", "h_2"]
        self.sampler = sampler
        self.ansatz = ansatz
        self.user_messenger = user_messenger
        self.prm = init_prm.copy()
        self.prm_jac = init_prm.copy()
        self.prm_threshold = 1e-12
        eval_op_idx = []
        vqd_cost_idx = [0]
        self.ovlp_ansatz = []
        self.ovlp_prm = []
        self.ovlp = []
        self.ovlp_name = []
        if if_vqd:
            for cost,val in vqd_set["cost"].items():
                if "overlap" not in cost:
                    try:
                        vqd_cost_idx.append(self.op_name.index(cost))
                    except:
                        self.user_messenger.publish(f"Operator {cost} does not exist.")
                else:
                    self.ovlp_name.append(cost)
                    self.ovlp_ansatz.append(val[0])
                    self.ovlp_prm.append(val[1])
                    self.ovlp.append(0.0)
        if if_vqeac:
            for const,val in vqeac_set["constraint"].items():
                if "overlap" not in const:
                    try:
                        eval_op_idx.append(self.op_name.index(const))
                    except:
                        self.user_messenger.publish(f"Operator {const} does not exist.")
                else:
                    self.ovlp_name.append(const)
                    self.ovlp_ansatz.append(val[0])
                    self.ovlp_prm.append(val[1])
                    self.ovlp.append(0.0)
        self.jac_set = jac_set
        self.vqd_cost_idx = vqd_cost_idx
        self.eval_op_idx = list(set(eval_op_idx + vqd_cost_idx))
        self.eval_op = [op for idx,op in enumerate(operators) if idx in self.eval_op_idx]
        self.calculate()
        self.expvals_jac = np.zeros((len(self.eval_op), len(self.prm)))
        self.expvals_jac_sample = np.zeros((len(self.eval_op), len(self.prm)))
        self.ovlp_jac = np.zeros((len(self.ovlp), len(self.prm)))
        self.init = True
    def calculate(self):
        self.expvals = calc_expvals(self.prm, self.sampler, self.ansatz, self.eval_op, 0, self.user_messenger)
        self.user_messenger.publish(f"prm= {self.prm} expvals= {self.expvals}")
        if len(self.ovlp) > 0:
            self.ovlp = np.array([
                calc_overlap(self.prm, p, self.ansatz, a, self.sampler, None, None, 0, self.user_messenger)
                for a,p in zip(self.ovlp_ansatz, self.ovlp_prm)
            ])
            self.ovlp = [(ov.conjugate() * ov).real for ov in self.ovlp]
            self.user_messenger.publish(f"prm= {self.prm} ovlps= {self.ovlp}")
        self.eval_count += 1
        self.init = False
    def calculate_jac(self):
        self.expvals_jac = calc_expvals_jac(self.prm_jac, self.jac_set, self.sampler, self.ansatz, self.eval_op, 0, self.user_messenger)
        self.user_messenger.publish(f"prm= {self.prm_jac} expvals_jac= {self.expvals_jac}")
        self.init = False
    def get_cost(self, p):
        if np.max(np.abs(p - self.prm)) > self.prm_threshold:
            self.prm = p.copy()
            self.calculate()
        res = []
        for op_idx in self.vqd_cost_idx:
            res.append(self.expvals[self.eval_op_idx.index(op_idx)])
        res += self.ovlp
        return np.array(res)
    def get_propaties(self, p, obsname):
        if np.max(np.abs(p - self.prm)) > self.prm_threshold:
            self.prm = p.copy()
            self.calculate()
        if "overlap" not in obsname:
            res = self.expvals[self.eval_op_idx.index(self.op_name.index(obsname))]
        else:
            res = self.ovlp[self.ovlp_name.index(obsname)]
        return res
    def get_cost_jac(self, p):
        if self.init or np.max(np.abs(p - self.prm_jac)) > self.prm_threshold:
            self.prm_jac = p.copy()
            self.calculate_jac()
        res = []
        for op_idx in self.vqd_cost_idx:
            res.append(self.expvals_jac[self.eval_op_idx.index(op_idx)])
        res += self.ovlp_jac
        return np.array(res)
    def get_propaties_jac(self, p, obsname):
        if self.init or np.max(np.abs(p - self.prm_jac)) > self.prm_threshold:
            self.prm_jac = p.copy()
            self.calculate_jac()
        res = self.expvals_jac[self.eval_op_idx.index(self.op_name.index(obsname))]
        return res
    def get_cost_jac_sample(self, p):
        self.expvals_jac_sample = calc_expvals_jac(
            p.copy(), self.jac_set, self.sampler, self.ansatz, self.eval_op, 0, self.user_messenger,
            previous_result=self.expvals_jac_sample
        )
        self.user_messenger.publish(f"prm= {self.prm_jac} expvals_jac(sample)= {self.expvals_jac_sample}")
        res = []
        for op_idx in self.vqd_cost_idx:
            res.append(self.expvals_jac_sample[self.eval_op_idx.index(op_idx)])
        return np.array(res)
    def get_propaties_jac_sample(self, p, obsname):
        self.expvals_jac_sample = calc_expvals_jac(
            p.copy(), self.jac_set, self.sampler, self.ansatz, self.eval_op, 0, self.user_messenger,
            previous_result=self.expvals_jac_sample
        )
        self.user_messenger.publish(f"prm= {self.prm_jac} expvals_jac(sample)= {self.expvals_jac_sample}")
        res = self.expvals_jac_sample[self.eval_op_idx.index(self.op_name.index(obsname))]
        return res



def state_tomography(ansatz, prm, sampler, ro_miti=False, seed=50, purify=True):
    """ execute quantum state-tomography.
    see https://qiskit.org/documentation/experiments/tutorials/state_tomography.html
    Args:
        ansatz(QuantumCircuit): ansatz circuit
        prm(np.array([float])): optimal parameters
        sampler(CircuitSampler): circuit sampler
        ro_miti(Bool): whether to execute readout mitigation
        seed(int): seed for simulation
        purify(Bool):  whether to execute purification
    """
    def get_calib(circ, sampler):
        meas_calibs, state_labels = complete_meas_cal(
            qubit_list=range(circ.num_qubits),
            qr = circ.qregs[0], # I do not know why [0] is necessary. It may be a bag.
            circlabel="mcal"
        )
        job_calib = execute(
            meas_calibs,
            backend=sampler.quantum_instance.backend,
            shots=sampler.quantum_instance.run_config.shots,
            initial_layout=sampler.quantum_instance.compile_config['initial_layout']
        )
        fitter = CompleteMeasFitter(job_calib.result(), state_labels, circlabel='mcal')
        return fitter

    bound_circ = ansatz.bind_parameters(prm)
    qst = StateTomography(bound_circ)
    qst_circ = qst.circuits()
    qst_res = execute(
        qst_circ,
        backend=sampler.quantum_instance.backend,
        shots=sampler.quantum_instance.run_config.shots,
        initial_layout=sampler.quantum_instance.compile_config['initial_layout']
    ).result()
    if ro_miti:
        calib_fitter = get_calib(ansatz, sampler)
        qst_res = calib_fitter.filter.apply(qst_res)
    qst_exp = ExperimentData()
    qst_exp.add_data(qst_res)
    # density_mat = qst.run_analysis(qst_exp).analysis_results("state").value
    density_mat = qst.analysis.run(qst_exp).analysis_results("state").value
        
    # diagonalize density_mat to get maximum eigenvalue and its eigenvector (purificatoin)
    eigenvals, eigenvecs = np.linalg.eig(density_mat)
    eigenvals = np.real(eigenvals)
    eigenvecs = np.transpose(eigenvecs)
    psi = eigenvecs[np.argmax(eigenvals)]
    return density_mat, psi, np.max(eigenvals)



def SPSA_wrapper(fun, x0, args, **kwargs):
    """ wrapper for SPSA optimizer implemented in Qiskit.
    Args:
        fun(callable): function to minimize.
        x0(np.array(float)): initial parameters.
        args(tuple, optional): extra arguments passed to the objective function.
        kwargs(dict): any other arguments passed to scipy.minimize and solver options.
    """

    maxiter = kwargs.get("maxiter", 100)
    tol = kwargs.get("tol", 1e-6)
    last_avg = kwargs.get("last_avg", 1)
    if_callback = kwargs.get("callback_SPSA", True)
    user_messenger = kwargs.get("user_messenger", None)

    vals = []
    def termination_checker(nfev, prm, fun, step, step_accept): # documentation is WRONG.
        vals.append(fun)
        if len(vals) > 5 and (max(vals[-5:]) - min(vals[-5:])) < tol: # check latest 5 results
            return True
        else:
            return False
    def callback(nfev, prm, fun, step, step_accept):
        if user_messenger is not None:
            user_messenger.publish(f"callback: nfev= {nfev}, prm= {prm}, fun= {fun}")

    spsa = SPSA(
        maxiter=maxiter,
        last_avg=last_avg,
        second_order=False if kwargs["jac"] is None else True,
        callback=callback if if_callback else None,
        termination_checker=termination_checker
    )
    res_tmp = spsa.minimize(fun=fun, x0=x0, jac=kwargs["jac"], bounds=kwargs.get("bounds", None))
    res = OptimizeResult(
        x=res_tmp.x, 
        success=True, 
        fun=res_tmp.fun, 
        jac=res_tmp.jac, 
        nfev=res_tmp.nfev, 
        njev=res_tmp.njev, 
        nit=res_tmp.nit
    )    
    return res



def gradient_descent(fun, x0, args, **kwargs):
    """ gradient descent optimizer. If combined with finite samplings, this becomes the stochastic gradient descent(?).
    Args:
        fun(callable): function to minimize. Not used in this optimizer.
        x0(np.array(float)): initial parameters.
        args(tuple, optional): extra arguments passed to the objective function.
        kwargs(dict): any other arguments passed to scipy.minimize.
    """
    maxiter = kwargs.get("maxiter", 100)
    tol_prm = kwargs.get("tol_prm", 1e-6)
    learn_rate = kwargs.get("learn_rate", 0.1)
    gtol = kwargs.get("gtol", 1e-6)
    jac = kwargs["jac"]

    prm = x0
    diff = np.inf; grad = np.inf
    for i in range(maxiter):
        if np.all(np.abs(diff) < tol_prm) or np.max(np.abs(grad)) < gtol:
            break
        grad = jac(prm)
        # print(f"debug, prm={prm}, grad= {grad}")
        diff = learn_rate * grad
        prm -= diff
    success = True if i+1 < maxiter else False 
    res = OptimizeResult(x=prm, success=success, fun=fun(prm), jac=grad, nfev=1, njev=i+1, nit=i+1)
    return res



def Adam(fun, x0, args, **kwargs):
    """ Adam optimizer. See https://arxiv.org/abs/1412.6980v8
    Args: 
        fun(callable): function to minimize. Not used in this optimizer.
        x0(np.array(float)): initial parameters.
        args(tuple, optional): extra arguments passed to the objective function.
        kwargs(dict): any other arguments passed to scipy.minimize.
    """
    maxiter = kwargs.get("maxiter", 100)
    alpha = kwargs.get("alpha", 0.001)
    beta1 = kwargs.get("beta1", 0.9)
    beta2 = kwargs.get("beta2", 0.999)
    epsilon = kwargs.get("epsilon", 1e-8)
    tol_prm = kwargs.get("tol_prm", 1e-6)
    gtol = kwargs.get("gtol", 1e-6)
    jac = kwargs["jac"]
    calculate_zeroth = kwargs.get("calculate_zeroth", False)
    
    prm = x0; m = 0.0; v = 0.0; t = 0 # initialize
    diff = np.inf; g = np.inf
    for i in range(maxiter):
        if np.all(np.abs(diff) < tol_prm) or np.max(np.abs(g)) < gtol:
            break
        t += 1
        if calculate_zeroth:
            f = fun(prm)
        g = jac(prm) # get gradients at timestep t
        m = beta1 * m + (1 - beta1) * g # update biased first moment estimate
        v = beta2 * v + (1 - beta2) * g**2 # update biased second raw moment estimate
        m_ = m / (1 - beta1**t) # compute bias-corrected first moment estimate
        v_ = v / (1 - beta2**t) # compute bias-corrected second raw moment estimate
        diff = alpha * m_ / (np.sqrt(v_) + epsilon) # update parameters
        prm -= diff
    success = True if i+1 < maxiter else False

    if calculate_zeroth:
        fun = f
        nfev = i + 1
    else:
        fun = fun(prm)
        nfev = 1
    res = OptimizeResult(x=prm, success=success, fun=fun, jac=g, nfev=nfev, njev=i+1, nit=i+1)
    return res



def execute_vqe(
    qubit_ops,
    sampler, 
    ansatz, 
    user_messenger, 
    init_prm,
    options,
    opt_options={},
    state_index=0,
):
    """ exicutes VQE or VQD or VQE/AC.
    Args:
        qubit_ops(list[list[(str,complex)]]): list of qubit operators to measure.
        sampler(CircuitSmapler): CircuitSmapler instance. 
        ansatz(QuantumCircuit): ansatz circuit with parameters not assigned.
        user_messenger(UserMessenger): Used to communicate with the program user.
        init_prm(np.array(float)): initial parameters.
        options(dict): options for optimization.
        opt_options(dict): options dict passed to the optimizer.
        state_index(int): 0 refres to the ground state.
    """
    user_messenger.publish(f"Optimization by {options['method']}")

    execute_vqd = options.get("execute_vqd", False)
    vqd_settings = options.get("vqd_settings", {"cost": {"overlap":(ansatz, np.zeros(ansatz.num_parameters), 0.0, 1.0), "ssq":(0.0, 1.0)}})
    execute_vqeac = options.get("execute_vqeac", False)
    vqeac_settings = options.get("vqeac_settings", {"constraint": {"overlap":(ansatz, np.zeros(ansatz.num_parameters), 0.0, 1e-3), "ssq":(0.0, 1e-3)}})
    jac_settings = options.get("jac_settings", {"shift":np.pi/2.0, "sample_rate":1.0})

    calculator = Calculator(
        qubit_ops,
        sampler,
        ansatz,
        user_messenger,
        init_prm,
        execute_vqd,
        vqd_settings,
        execute_vqeac,
        vqeac_settings,
        jac_settings,
    )

    constraints = []
    if execute_vqeac:
        if state_index != 0: # excited states
            pass # TODO, calculate overlap
        for key,val in vqeac_settings["constraint"].items():
            if "overlap" not in key:
                constraints.append({
                    "type":"ineq",
                    "fun":lambda x: val[1] - abs(calculator.get_propaties(x, key) - val[0])
                })
            else:
                constraints.append({
                    "type":"ineq",
                    "fun":lambda x: val[3] - abs(calculator.get_propaties(x, key) - val[2])
                })

    threshold = 1e-12
    cost_target = np.array(
        [0.0] + # for hamiltonian
        [val[0] for key,val in vqd_settings["cost"].items() if "overlap" not in key] + # for panelty
        [val[2] for key,val in vqd_settings["cost"].items() if "overlap" in key] # for overlap
    )
    cost_weights = np.array(
        [1.0] + # for hamiltonian
        [val[1] for key,val in vqd_settings["cost"].items() if "overlap" not in key] + # for panelty
        [val[3] for key,val in vqd_settings["cost"].items() if "overlap" in key] # for overlap
    )
    cost_notes = (
        [""] + 
        [val[2] if len(val) > 2 else "" for key,val in vqd_settings["cost"].items() if "overlap" not in key] + 
        [val[4] if len(val) > 4 else "" for key,val in vqd_settings["cost"].items() if "overlap" in key]
    )
    tgn0_idxs = [ic for ic,c in enumerate(cost_target) if abs(c) > threshold] # index of cost target != 0
    square_idxs = [ic for ic,c in enumerate(cost_notes) if "squared" in c.split("_")] # index of cost <O>^2
    weight_inverses = [ic for ic,c in enumerate(cost_notes) if "inverse" in c.split("_")]
    weight_cutoffs = [[c_ for c_ in c.split("_") if "cutoff-" in c_][0] if "cutoff-" in c else 0 for c in cost_notes]
    weight_cutoffs = [int(wco.replace("cutoff-","")) if type(wco) is str else wco for wco in weight_cutoffs]
    use_jac = options.get("use_jac", False)

    if options["method"] == "SPSA":
        opt_method = SPSA_wrapper
        opt_options["user_messenger"] = user_messenger
    elif options["method"] == "gradient-descent":
        opt_method = gradient_descent
        use_jac = True
        opt_options["user_messenger"] = user_messenger
    elif options["method"] == "Adam":
        opt_method = Adam
        use_jac = True
        opt_options["user_messenger"] = user_messenger
    else:
        opt_method = options["method"]

    # function callables
    if execute_vqd:
        # func = lambda x: sum((calculator.get_cost(x) - cost_target) * cost_weights)
        def func(x):
            cost = calculator.get_cost(x)
            # print(f"debug, cost= {cost}")
            eval_ct = calculator.eval_count
            if eval_ct > ansatz.num_parameters:
                eval_ct -= ansatz.num_parameters
            else:
                eval_ct = 1
            cost_terms = []
            cost_weights_tmp = []
            for c_idx,c in enumerate(cost):
                if c_idx in square_idxs:
                    cost_terms.append((c - cost_target[c_idx])**2)
                else:
                    if c_idx in tgn0_idxs:
                        cost_terms.append(abs(c - cost_target[c_idx]))
                    else:
                        cost_terms.append(c)
                weight_tmp = cost_weights[c_idx]
                if c_idx in weight_inverses:
                    weight_tmp = weight_tmp / eval_ct
                if weight_cutoffs[c_idx] != 0:
                    if eval_ct > weight_cutoffs[c_idx]:
                        weight_tmp = 0
                cost_weights_tmp.append(weight_tmp)
            # print(f"debug, cost_terms= {cost_terms}")
            # print(f"debug, cost_weights_tmp= {cost_weights_tmp}")
            cost = sum(np.array(cost_terms) * np.array(cost_weights_tmp))
            # print(f"debug, cost= {cost}")
            return cost
    else:
        func = lambda x: calculator.get_propaties(x, "ene")
    if use_jac:
        if execute_vqd:
            tgn0_idx = [ic for ic,c in enumerate(cost_target) if abs(c) > threshold] # index of cost target != 0
            square_idx = sorted(list(set(
                [ic for ic,c in enumerate(cost_notes) if (c is not None) and ('squared' in c)] + tgn0_idx
            ))) # add penalty terms with the form of <O>^2.
            if len(square_idx) == 0:
                if abs(1.0 - jac_settings.get("sample_rate", 1.0)) > threshold:
                    jac_func = lambda x: sum([co * ja for co,ja in zip(cost_weights,calculator.get_cost_jac_sample(x))])
                else:
                    jac_func = lambda x: sum([co * ja for co,ja in zip(cost_weights,calculator.get_cost_jac(x))])
            else:
                def jac_func(x):
                    if abs(1.0 - jac_settings.get("sample_rate", 1.0)) > threshold:
                        jac = calculator.get_cost_jac_sample(x)
                    else:
                        jac = calculator.get_cost_jac(x)
                    cost = calculator.get_cost(x)
                    jac_terms = [(cost[ija] - cost_target[ija]) * 2 * ja if ija in square_idx else ja for ija,ja in enumerate(jac)]
                    jac = sum([ co * ja for co,ja in zip(cost_weights, jac_terms)])
                    return jac
        else:
            if abs(1.0 - jac_settings.get("sample_rate", 1.0)) > threshold:
                jac_func = lambda x: calculator.get_propaties_jac_sample(x, "ene")
            else:
                jac_func = lambda x: calculator.get_propaties_jac(x, "ene")
    else:
        jac_func = None

    res = minimize(
        fun=func,
        x0=init_prm,
        method=opt_method,
        jac=jac_func,
        bounds=options.get("bounds", None),
        constraints=constraints,
        callback=None, #TODO, callback function
        options=opt_options 
    )

    user_messenger.publish("optimization finished.")
    for k,v in zip(res.keys(), res.values()):
        user_messenger.publish(f"{k:12}: {v}")

    fargs_tmp = (sampler, ansatz, qubit_ops[1:], 0, user_messenger)
    aux_op = calc_expvals(res.x, *fargs_tmp)
    user_messenger.publish(f"aux-op      : {aux_op}")
    return res



def main(backend, user_messenger, **kwargs):
    """ entry function.
    Args:
        backend(ProgramBackend): Backend for the circuits to run on.
        user_messenger(UserMessenger): Used to communicate with the program user.
        kwargs: User inputs.
            qubit_ops(list[list[(str,complex)]]): list of qubit operators to measure.
                list of tuples consist of Pauli-label(e.g. "XX") and its coefficient.
            ansatz(QuauntumCircuit): ansatz circuit
            # ansatz(str): ansatz name
            init_prm(np.array(float)): initial parameters
            options(dict): settings for measurement
            opt_options(dict): setting for optimization
    """
    time_s = time.time()
    np.set_printoptions(threshold=10000, linewidth=10000)

    ## settings
    seed = kwargs["options"].get("seed", 50)
    np.random.seed(seed)
    random.seed(seed)
    algorithm_globals.random_seed = seed
    if backend is not None:
        quantum_instance = QuantumInstance(
            backend=backend,
            shots=kwargs["options"].get("shots", 1024),
            seed_simulator=seed,
            initial_layout=kwargs["options"].get("initial_layout", None),
            seed_transpiler=seed,
            measurement_error_mitigation_cls=(
                CompleteMeasFitter if kwargs["options"].get("readout_mitigation", False) else None
            ),
            cals_matrix_refresh_period=kwargs["options"].get("readout_mitigation_period", 10),
        )
        sampler = CircuitSampler(
            backend=quantum_instance,
            statevector=True if type(backend) is StatevectorSimulator else False
        )
    else:
        sampler = None
    state_index = kwargs.get("state_index", 0)
    if kwargs["options"].get("exclude_identity", False):
        shift = [qop[0][1] if qop[0][0]=="I"*len(qop[0][0]) else 0.0 for qop in kwargs["qubit_ops"]]
        user_messenger.publish(f"Identity term is excluded from the hamiltonian: shift= {shift[0].real}")
        # qubit_ops = [[p for p in qop if p[0]!="I"*len(p[0])] for qop in kwargs["qubit_ops"]]
        qubit_ops = kwargs["qubit_ops"].copy()
        qubit_ops[0] = [p for p in qubit_ops[0] if p[0]!="I"*len(p[0])] # currently only Hamiltonian is modified.
    else:
        qubit_ops = kwargs["qubit_ops"]

    res = execute_vqe(
        qubit_ops,
        sampler,
        kwargs["ansatz"],
        user_messenger,
        kwargs.get("init_prm", np.ones(kwargs["ansatz"].num_parameters)),
        kwargs["options"],
        kwargs.get("opt_options", {"tol":1e-4, "maxiter":100}),
        kwargs.get("state_index", 0),
    )

    if kwargs["options"].get("state_tomo", False):
        density_mat, psi, eigval = state_tomography(
            kwargs["ansatz"],
            res.x.copy(),
            sampler,
            ro_miti=kwargs["options"].get("ro_miti_for_st", False),
            seed=seed,
            purify=True
        )
        user_messenger.publish(f"state-tomography: density matrix(eigval={eigval})= {density_mat}")
        if kwargs["ansatz"].num_qubits > 16:
            mat_ops = [PauliSumOp.from_list(op).to_matrix(massive=True) for op in qubit_ops]
        else:
            mat_ops = [PauliSumOp.from_list(op).to_matrix() for op in qubit_ops]
        vals = [np.real(np.conjugate(psi).dot(m_op.dot(psi))) for m_op in mat_ops]
        user_messenger.publish(f"state-tomography and purified result: {vals}")

    # second optimization 
    if "options_2" in kwargs:
        if backend is not None:
            quantum_instance_2 = QuantumInstance(
                backend=backend,
                shots=kwargs["options_2"].get("shots", 1024),
                seed_simulator=seed,
                initial_layout=kwargs["options_2"].get("initial_layout", None),
                seed_transpiler=seed,
                measurement_error_mitigation_cls=(
                    CompleteMeasFitter if kwargs["options_2"].get("readout_mitigation", False) else None
                ),
                cals_matrix_refresh_period=kwargs["options_2"].get("readout_mitigation_period", 10),
            )
            sampler_2 = CircuitSampler(
                backend=quantum_instance_2,
                statevector=True if type(backend) is StatevectorSimulator else False
            )
        else:
            sampler_2 = None
        res_2 = execute_vqe(
            qubit_ops,
            sampler_2,
            kwargs["ansatz"],
            user_messenger,
            res.x.copy(),
            kwargs["options_2"],
            kwargs.get("opt_options_2", kwargs.get("opt_options",{"tol":1e-4, "maxiter":100})),
            kwargs.get("state_index", 0),
        )
    
        if kwargs["options_2"].get("state_tomo", False):
            density_mat, psi, eigval = state_tomography(
                kwargs["ansatz"],
                res_2.x.copy(),
                sampler_2,
                ro_miti=kwargs["options_2"].get("ro_miti_for_st", False),
                seed=seed,
                purify=True
            )
            user_messenger.publish(f"state-tomography(2): density matrix(eigval={eigval})= {density_mat}")
            if kwargs["ansatz"].num_qubits > 16:
                mat_ops = [PauliSumOp.from_list(op).to_matrix(massive=True) for op in qubit_ops]
            else:
                mat_ops = [PauliSumOp.from_list(op).to_matrix() for op in qubit_ops]
            vals = [np.real(np.conjugate(psi).dot(m_op.dot(psi))) for m_op in mat_ops]
            user_messenger.publish(f"state-tomography and purified result(2): {vals}")

    time_e = time.time() - time_s
    user_messenger.publish(f"Elapsed time: {time_e} s")

    res_list = [dict(zip(res.keys(), [bool(v) if type(v)==np.bool_ else v for v in res.values()]))]
    if "options_2" in kwargs:
        res_list.append(dict(zip(res_2.keys(), [bool(v) if type(v)==np.bool_ else v for v in res_2.values()])))
    return res_list



if __name__ == "__main__":
    from qiskit_ibm_runtime.program import UserMessenger
    from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder
    import json
    # import make_operator

    if 0: # Aer simulator for local test
        from qiskit import Aer
        # backend = Aer.get_backend("statevector_simulator") # INfinite sampling 
        backend = Aer.get_backend("aer_simulator_statevector") # finite sampling
    if 0: # from saved noisemodel
        import pickle
        with open("/home/gocho/qiskit/gradient_calibration/get_noise/kawasaki_backend.dat", "rb") as f:
            backend = pickle.load(f)
    backend = None

    user_messenger = UserMessenger()

    # CAS
    norb = 2
    nelec = 2

    if 0:
        # define molecule
        from pyscf import gto, scf
        mol = gto.M(
            atom='''
                C         -0.000000000000          0.000000000000         -0.005302010370
                C          0.000000000000          0.000000000000          1.335460010375
                H          0.000000000000          0.918153736897         -0.576538208909
                H         -0.000000000000         -0.918153736897         -0.576538208909
                H         -0.000000000000          0.918153736899          1.906696208906
                H          0.000000000000         -0.918153736899          1.906696208906''',
            basis="sto3g",
            symmetry=False
        )
        # execute classical HF calculation
        mf = scf.RHF(mol)
        mf.kernel()
        ferops = make_operator.get_ferop(mf, norb, nelec)
        from qiskit_nature.mappers.second_quantization import ParityMapper
        mapper = ParityMapper()
        qubit_ops = [make_operator.get_qubitop(fop, mapper, nelec, True) for fop in ferops]
        qubit_ops = [qop.primitive.to_list() for qop in qubit_ops]
    if 1:
        qubit_ops = [
            [
                ('II', (-76.76631916830594+0j)),
                ('IZ', (0.1545406433989221+0j)),
                ('ZI', (-0.15454064339892248+0j)),
                ('ZZ', (-0.00329890469082067+0j)),
                ('XX', (0.17216032212807028+0j))
            ],
            [
                ('II', (1.9999999999999993+0j))
            ],
            [
                ('II', (0.5000000000000003+0j)),
                ('ZZ', (0.49999999999999706+0j)),
                ('YY', (0.4999999999999999+0j)),
                ('XX', (-0.49999999999999994+0j))
            ],
            [
                ('II', 0j)
            ],
            [
                ('II', (5893.145174330428+4.547473508864641e-13j)),
                ('IZ', (-23.726013081567196+0j)),
                ('ZI', (23.72601308156723+0j)),
                ('ZZ', (0.45872391987841965+0j)),
                ('YY', (0.001135880988483606+0j)),
                ('XX', (-26.43222847320362+0j))
            ]
        ]


    if 0: # spin-restricted ansatz
        assert len(qubit_ops[0][0]) == 2, "Number of qubits does not match to the selected ansatz"
        from qiskit.circuit import ParameterVector, QuantumRegister, QuantumCircuit
        theta = ParameterVector('θ', 2)
        ansatz = QuantumCircuit(QuantumRegister(2, 'q'))
        ansatz.u(theta[0], 0, 0, 0)  # RY(theta)
        ansatz.x(1)
        ansatz.cx(0, 1)
        ansatz.u(theta[1], 0, 0, 0)
        ansatz.u(-theta[1], 0, 0, 1)
    if 0: #real-amplitudes
        from qiskit.circuit.library import RealAmplitudes
        ansatz = RealAmplitudes(len(qubit_ops[0][0][0]), entanglement="linear", reps=3, parameter_prefix="p")
    if 1: # real-amplitudes
        from qiskit.circuit.library import RealAmplitudes
        ansatz = RealAmplitudes(len(qubit_ops[0][0][0]), entanglement="full", reps=1)


    init_prm = np.ones(ansatz.num_parameters)
    # init_prm = np.array([0, np.pi])

    inputs = {}
    inputs["qubit_ops"] = qubit_ops
    inputs["ansatz"] = ansatz
    inputs["init_prm"] = init_prm
    inputs["options"] = {
        "shots":8192,
        "method":"COBYLA",
        "readout_mitigation":False,
        "seed": 50,
        "state_tomo":False,
        "ro_miti_for_st":False,
        "execute_vqd":True,
        "vqd_settings":{"cost": {"overlap":(ansatz, np.zeros(ansatz.num_parameters), 0.0, 1.0, "cutoff-10"), "ssq":(0.0, 1.0)}}
    }
    inputs["opt_options"] = {"tol":1e-4, "maxiter":100}

    serialized_inputs = json.dumps(inputs, cls=RuntimeEncoder)
    deserialized_inputs = json.loads(serialized_inputs, cls=RuntimeDecoder)

    res = main(backend, user_messenger, **deserialized_inputs)
    # print(f"VQE result= {res}")


