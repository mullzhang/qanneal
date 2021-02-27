from itertools import combinations
import math

from qutip import sigmax, sigmaz, qeye, tensor, Qobj, sesolve, ket2dm
import dimod
import numpy as np


class IsingHamiltonian:
    def __init__(self, h, J):
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
        self.num_spins = bqm.num_variables

        self.pauliz = [tensor([sigmaz() if i == j else qeye(2) for j in range(self.num_spins)])
                       for i in range(self.num_spins)]

        self.H_prob = sum(v * self.pauliz[i] for i, v in h.items())
        self.H_prob += sum(v * self.pauliz[i] * self.pauliz[j] for (i, j), v in J.items())
        self.evals_prob, self.ekets_prob = self.H_prob.eigenstates()

        self.paulix = None
        self.H_driver = 0

    @staticmethod
    def linear_inc(t, args):
        return t / args['annealing_time']

    @staticmethod
    def linear_dec(t, args):
        return 1 - t / args['annealing_time']

    def _set_paulix(self):
        if self.paulix is None:
            self.paulix = [tensor([sigmax() if i == j else qeye(2) for j in range(self.num_spins)])
                           for i in range(self.num_spins)]

    def induce_transverse_field(self):
        self._set_paulix()
        self.H_driver += -sum(self.paulix)

    def induce_highord_driver(self, n):
        self._set_paulix()
        self.H_driver += -sum(math.prod(sx) for sx in combinations(self.paulix, n))

    def _gen_hamil_list(self, sched_prob, sched_driver):
        if sched_prob is None:
            sched_prob = IsingHamiltonian.linear_inc
        if sched_driver is None:
            sched_driver = IsingHamiltonian.linear_dec

        return [[self.H_prob, sched_prob], [self.H_driver, sched_driver]]

    def energy_spectrum(self, tlist, sched_prob=None, sched_driver=None):
        Hlist = self._gen_hamil_list(sched_prob, sched_driver)

        annealing_time = tlist[-1]
        _args = {'annealing_time': annealing_time}

        Elists = []
        for t in tlist:
            H = sum(hamil * sched(t, _args) for hamil, sched in Hlist)
            evals = H.eigenenergies()
            Elists.append((t, [evals[i] for i in range(2**self.num_spins)]))

        arr = np.array(Elists, dtype=[('time', float), ('energy', (float, 2**self.num_spins))])
        return arr.view(np.recarray)

    def solve_sdeq(self, tlist, e_ops=None, sched_prob=None, sched_driver=None, **sched_args):
        annealing_time = tlist[-1]
        evals_driver, ekets_driver = self.H_driver.eigenstates()
        psi0 = ekets_driver[np.argmin(evals_driver)]

        if e_ops is None:
            e_ops = [ket2dm(ek) for ek in self.ekets_prob]

        Hlist = self._gen_hamil_list(sched_prob, sched_driver)

        args = sched_args.copy()
        args.update(dict(annealing_time=annealing_time))
        result = sesolve(Hlist, psi0, tlist, e_ops=e_ops, args=args)
        expects = np.transpose(result.expect)

        arr = np.array(list(zip(tlist, expects)), dtype=[('time', float), ('expect', (float, len(expects[0])))])
        return arr.view(np.recarray)
