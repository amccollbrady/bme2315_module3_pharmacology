import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Cavalitide parameters
Favailable = 0.01
ka = 0.006          # 1/h
D = 20.8            # mg per dose
interval = 24       # h (once daily)
n_doses = 100

# Shared model parameters
ke = 0.0039         # 1/h
V = 10.8            # L
MW = 4113.6         # g/mol
Ffree = 0.01

def ODEfunc(t, y):
    Asc, Cc = y
    return [-ka * Asc, ka * Asc / V - ke * Cc]

def simulate(D, n_doses, interval, dt=0.1):
    t_out, Cc_out = [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))
    for i in range(n_doses):
        y[0] += Favailable * D
        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)
        solution = solve_ivp(ODEfunc, (t_seg[0], t_seg[-1]), y,
                             t_eval=t_seg, method="RK45",
                             rtol=1e-8, atol=1e-10)
        if not solution.success:
            raise RuntimeError(solution.message)
        t_out.append(solution.t)
        Cc_out.append(solution.y[1])
        y = solution.y[:, -1].copy()
    t = np.concatenate(t_out)
    Cc = np.concatenate(Cc_out)
    free_Cc_nM = Ffree * Cc * 1e6 / MW
    return t / 24, free_Cc_nM

time_days, free_concentration = simulate(D, n_doses, interval)
final_interval = time_days >= (n_doses - 1) * interval / 24
steady_state_average = np.mean(free_concentration[final_interval])
print(f"Average concentration over final 24 h: {steady_state_average:.3f} nM")

plt.figure(figsize=(9, 5))
plt.plot(time_days, free_concentration, label=f"Cavalitide: {D:.1f} mg once daily")
plt.axhline(0.5, linestyle="--", label="Target = 0.5 nM")
plt.xlabel("Time (days)")
plt.ylabel("Free semaglutide concentration (nM)")
plt.title("Optimized Cavalitide Daily Dosing Profile")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Cavalitide_optimized_dosing.png", dpi=300, bbox_inches="tight")
plt.show()