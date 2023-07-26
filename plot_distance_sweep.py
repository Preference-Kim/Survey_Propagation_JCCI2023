import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, IndexLocator, FuncFormatter
from matplotlib.dates import MonthLocator, DateFormatter
#plt.style.use(['science','ieee'])

plt.rcParams['pdf.fonttype'] = 42 # TrueType
plt.rcParams['ps.fonttype'] = 42 

N_AP=300
N_USER=21
N_PILOT=7
SEED=100

def main():

    print("working")
    success = np.loadtxt(f"sumrates_SP_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")
    idx_success = np.argsort(success)

    mean_distance = np.loadtxt(f"mean_distance_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")
    mean_distance_s = np.loadtxt(f"mean_distance(successed)_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.csv")

    plot_d(mean_distance, 'mean_distance') #namestr(mean_distance,globals())
    plot_d(mean_distance_s, 'mean_distance_s')
    sorted_mean_distance_s = mean_distance_s[idx_success]
    plot_d(sorted_mean_distance_s, 'sorted_mean_distance_s')
    print("done")


def plot_d(x, x_name): 
    _, axes = plt.subplots(nrows=1, ncols=1,
                           tight_layout=True)
    
    n_trial = np.size(x)
    cdf_bins = np.arange(n_trial)/float(n_trial-1)
    axes.set_xticks(np.linspace(0, 1, 11)) #axes.set_yticks(np.linspace(0, 1, 11))
    axes.plot(cdf_bins, x, 'bo', label="Distance")
    axes.yaxis.set_major_locator(MultipleLocator(5)) ## x값이 5의 배수일 때마다 메인 눈금 표시
    axes.yaxis.set_major_formatter('{x:.0f}') ## 메인 눈금이 표시될 형식 
    axes.yaxis.set_minor_locator(MultipleLocator(1)) ## 서브 눈금은 x값이 1의 배수인 경우마다 표시
    axes.set_xlim(0, 1)
    axes.set_xlabel("Cases")
    axes.set_ylabel("Mean distance(m)")
    axes.grid(which='minor', alpha=0.5)
    axes.grid(which='major', alpha=1)

    plt.axhline(np.mean(x), color='red', label = "mean", linewidth=1.5)
    axes.legend() 

    #x_name = namestr(x,globals())
    #print(f"{x_name}")
    plt.savefig(f"{x_name}_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.pdf")
    plt.savefig(f"{x_name}_{N_AP}.{N_USER}.{N_PILOT}.seed{SEED}.png")

    return

if __name__=="__main__":
    main()