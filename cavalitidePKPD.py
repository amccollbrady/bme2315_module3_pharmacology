import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Cavalitide pharmacokinetic parameters (from Task 2)
Favailable = 0.01
ka = 0.006          # 1/h
D = 20.8            # mg per daily dose, optimized in Task 2
interval = 24       # h
n_doses = 100       # 100 days

# Shared PK parameters
ke = 0.0039         # 1/h
V = 10.8            # L
MW = 4113.6         # g/mol
Ffree = 0.01        # fraction unbound in blood

# GLP1 receptor pharmacodynamic parameters (from Task 1)
Kd1 = 0.17          # nM, GLP-1 dissociation constant
Kd2 = 0.077         # nM, optimized Cavalitide dissociation constant
L1 = 0.015          # nM, fasting GLP-1 concentration

def ODEfunc(t, y):
    """PK model: depot amount (mg) and blood concentration (mg/L)."""
    Asc, Cc = y
    return [-ka * Asc, ka * Asc / V - ke * Cc]

def Fact(L1, L2, Kd1, Kd2):
    """Fraction of GLP1R receptors activated by GLP-1 and Cavalitide."""
    return (Kd2 * L1 + Kd1 * L2) / (
        Kd1 * Kd2 + Kd2 * L1 + Kd1 * L2
    )

def simulate_pkpd():
    y = np.array([0.0, 0.0])
    t_out, active_out = [], []
    dt = 0.1

    for i in range(n_doses):
        y[0] += Favailable * D
        t0, t1 = i * interval, (i + 1) * interval
        t_eval = t0 + dt * np.arange(int(round(interval / dt)) + 1)

        solution = solve_ivp(
            ODEfunc, (t0, t1), y, t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        start = 0 if i == 0 else 1
        t_out.append(solution.t[start:])
        total_nM = solution.y[1, start:] * 1e6 / MW
        free_nM = Ffree * total_nM
        active_out.append(Fact(L1, free_nM, Kd1, Kd2))
        y = solution.y[:, -1].copy()

    return np.concatenate(t_out) / 24, np.concatenate(active_out)

if __name__ == "__main__":
    time_days, fraction_active = simulate_pkpd()
    final_day = time_days >= 99
    print(f"Optimized Cavalitide Kd2: {Kd2:.3f} nM")
    print(f"Mean GLP1R activation on day 100: "
          f"{np.mean(fraction_active[final_day]):.4f}")
    print(f"Day 100 activation range: "
          f"{np.min(fraction_active[final_day]):.4f}–"
          f"{np.max(fraction_active[final_day]):.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(time_days, fraction_active, label="GLP1R fraction active")
    plt.xlabel("Time (days)")
    plt.ylabel("Fraction of active GLP1R receptors")
    plt.title("Predicted GLP1R Activation During Cavalitide Treatment")
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cavalitidePKPD_receptor_activation.png",
                dpi=300, bbox_inches="tight")
    plt.show()