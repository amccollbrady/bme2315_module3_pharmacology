# ligand_receptor.py

import numpy as np
import matplotlib.pyplot as plt

def Fact(G, L, Kd1, Kd2):
    """Fractional receptor activation with co-stimulation of GLP1 (G) and Semaglutide (L)"""
    return (Kd2 * G + Kd1 * L) / (Kd1 * Kd2 + Kd2 * G + Kd1 * L) # equation from lecture for two ligands competing for the receptor

Kd1 = 0.17 # dissociation constant for GLP1 [nM]
Kd2 = 0.38 # dissociation constant for semaglutide [nM]

G_fasting = 0.015 # fasting [GLP1] level [nM]
G_fed = 0.040 # post-feeding [GLP1] level [nM]

L = np.logspace(-3, 2, 500) # semaglutide concentration, 0.001 to 100 [nM]

plt.figure() # create a new figure window for plotting
plt.semilogx(L, Fact(G_fasting, L, Kd1, Kd2), label = "Fasting [GLP1] = 0.015 nM", lw=2) # fasting curve, semiglutide x & fraction active y
plt.semilogx(L, Fact(G_fed, L, Kd1, Kd2), label = "Fed [GLP1] = 0.040 nM", lw=2) # post-feeding curve, semaglutide x & fraction active y
plt.axhline(0.5, ls="--", c="gray", lw=1) # dashed reference line at 0.5
plt.ylim(0, 1) # set y-axis limits from 0 to 1
plt.xlabel("[Semaglutide] (nM, log scale)") # label x-axis
plt.ylabel("Fraction active (Fact)") # label y-axis
plt.title("GLP1R Co-stimulation") # title of the plot
plt.legend(loc="lower right") # legend in the lower right corner
plt.grid(True, which="both", ls="--", alpha=0.5) # background grid lines
plt.show() # show plot

# The active fraction for the fed levels of GLP1 is higher than the fasting levels of GLP1 at low semaglutide concentrations, 
# but as semaglutide concentrations increase, both the fed and fasting levels of GLP1 get closer until they converge as the drug 
# occupies the receptor.