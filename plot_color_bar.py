from xgutils import visutil
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_colorbar():
    fig, ax = visutil.newPlot()
    fraction = 1  # .05
    norm = mpl.colors.Normalize(vmin=-6, vmax=-1)
    cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='jet'),
                ax=ax, pad=.05, fraction=fraction,
                format=mpl.ticker.FuncFormatter(lambda x,pos:'$10^{%d}$'%x))
    ax.axis('off')
    # save as pdf
    plt.savefig(f"error_maps/colorbar.pdf", bbox_inches='tight')
    plt.savefig(f"error_maps/colorbar.png", bbox_inches='tight')

# Plot and save the colorbar
print("Save color bar...")
plot_colorbar()
