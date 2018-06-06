"""
Plotting functions used in attack.py
"""


import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm


rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True
rcParams['font.family'] = "serif"
rcParams['font.serif'] = "cm"


# # Plots an image (Variable)
# def plot_image(image):
#     plt.imshow(image.data.view(28, 28).numpy(), cmap='gray')


# # Plots and saves the comparison graph of an adversarial image
# def attack_result(model_name, p,
#                   img, img_pred, img_conf,
#                   adv, adv_pred, adv_conf):
#     model_name = model_name.replace('_', '\\_')  # Escapes '_' characters
#     r = (adv - img)
#     norm = r.norm(p)
#     # Matplotlib settings
#     rcParams['axes.titlepad'] = 10
#     rcParams['font.size'] = 8
#     fig = plt.figure(figsize=(7, 2.5), dpi=180)
#     # Image
#     ax1 = fig.add_subplot(1, 3, 1)
#     ax1.imshow(img.data.view(28, 28).cpu().numpy(), cmap='gray')
#     plt.title(f"\\texttt{{{model_name}(img)}} = {img_pred} \\small{{({100*img_conf:0.0f}\\%)}}")
#     plt.axis('off')
#     # Perturbation
#     ax2 = fig.add_subplot(1, 3, 2)
#     ax2.imshow(r.data.view(28, 28).cpu().numpy(), cmap='RdBu')
#     plt.title(f"Perturbation : $\\Vert r \\Vert_{{{p}}} = {norm:0.4f}$")
#     plt.axis('off')
#     # Adversarial image
#     ax3 = fig.add_subplot(1, 3, 3)
#     ax3.imshow(adv.data.view(28, 28).cpu().numpy(), cmap='gray')
#     plt.title(f"\\texttt{{{model_name}(img+r)}} = {adv_pred} \\small{{({100*adv_conf:0.0f}\\%)}}")
#     plt.axis('off')
#     # Save and plot
#     fig.tight_layout(pad=1)
#     plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.05)
#     plt.savefig("../results/latest/attack_result.png", transparent=True)


# Plots the history of a model training
def train_history(train_accs, test_accs):
    rcParams['font.size'] = 12
    t = list(range(len(train_accs)))
    plt.plot(t, train_accs, 'r')
    plt.plot(t, test_accs, 'b')
    plt.title("Network training history")
    plt.legend(["train accuracy", "test accuracy"])


# # Plots the history of an attack
# def attack_history(norms, confs):
#     rcParams['font.size'] = 14
#     t = list(range(len(norms)))
#     plt.plot(t, norms, 'r')
#     plt.plot(t, confs, 'b')
#     plt.legend(["$\\Vert r \\Vert$", "$\\mathrm{Conf}_c$"])
#     plt.savefig("../results/latest/attack_history.png", transparent=True)


# # Plots the space exploration
# def space_exploration(a, b, f):
#     T = [i/100 for i in range(-50, 151)]
#     X = [(1-t)*a + t*b for t in T]
#     Y = []
#     for x in tqdm(X):
#         Y.append(f(x))
#     plt.plot(T, Y)
#     plt.plot([0], Y[50], marker = 'o')
#     plt.plot([1], Y[150], marker = 'o')
# plt.savefig("../results/latest/space_exploration.png", transparent=True)