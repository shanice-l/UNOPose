def plot_3d(pcl1, pcl2, title, save_path=None):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection="3d"), figsize=plt.figaspect(1 / 3))
    axs[0].scatter(pcl1[:, 0], pcl1[:, 1], pcl1[:, 2], marker="o")
    axs[0].scatter(pcl2[:, 0], pcl2[:, 1], pcl2[:, 2], marker="x", color="red")

    axs[1].scatter(pcl1[:, 0], pcl1[:, 1], pcl1[:, 2], marker="o")
    axs[2].scatter(pcl2[:, 0], pcl2[:, 1], pcl2[:, 2], marker="x", color="red")

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
