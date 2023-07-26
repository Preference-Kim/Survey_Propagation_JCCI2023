import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, IndexLocator, FuncFormatter
from matplotlib.dates import MonthLocator, DateFormatter
#plt.style.use(['science','ieee'])

plt.rcParams['pdf.fonttype'] = 42 # TrueType
plt.rcParams['ps.fonttype'] = 42 

def main():

    _, axes = plt.subplots(nrows=1, ncols=1,
                           tight_layout=True)
    n_user=21
    n_ap=300
    n_pilot=7
    n_seed=100

    sumrates = np.loadtxt(f"sumrates_SP_{n_ap}.{n_user}.{n_pilot}.seed{n_seed}.csv")
    sumrates_sorted = np.sort(sumrates[~np.isnan(sumrates)])

    n_trial = np.size(sumrates_sorted)
    cdf_bins = np.arange(n_trial)/float(n_trial-1)

    ##
    #x = np.linspace(np.min(sumrates_sorted), np.max(sumrates_sorted))
    #dx = x[1]-x[0]
    #sumrates_diff = np.diff(sumrates_sorted)
    #pdf_bins = np.reciprocal(sumrates_diff)/n_trial

    axes.set_yticks(np.linspace(0, 1, 11))
    axes.plot(sumrates_sorted, cdf_bins, color='black', label="cdf") #
    ##
    ##plt.hist(sumrates_sorted, density = True, fc="none", ec="grey", label="frequency")
    #plt.plot(sumrates_sorted[1:], pdf_bins, color='red', label="pdf")

    #major_ticks = [700, 720, 740, 760, 780]
    #minor_ticks = [710, 730, 750, 770]
    #axes.set_xticks(major_ticks)
    #axes.set_xticks(minor_ticks, minor=True)
    #axes.set_xlim(min(major_ticks), max(minor_ticks))
    axes.xaxis.set_major_locator(MultipleLocator(5)) ## x값이 5의 배수일 때마다 메인 눈금 표시
    axes.xaxis.set_major_formatter('{x:.0f}') ## 메인 눈금이 표시될 형식 
    axes.xaxis.set_minor_locator(MultipleLocator(1)) ## 서브 눈금은 x값이 1의 배수인 경우마다 표시

    axes.set_ylim(0, 1)
    axes.set_xlabel("System throughput [Mbits/sec]")
    axes.set_ylabel("CDF")
    axes.grid(which='minor', alpha=0.5)
    axes.grid(which='major', alpha=1)

    plt.axvline(np.mean(sumrates_sorted), color='red', label = "mean", linewidth=1.5)
    #plt.axvline(760.71, color='orange', label = "chosen", linewidth=1.5)
    #plt.axvline(760.66, color='blue', label = "Hungarian", linestyle='--')
    #plt.axvline(718.15, color='green', label = "greedy", linestyle='--')
    #plt.axvline(730.48, color='yellow', label = "K-means", linestyle='--')

    axes.legend() 

    # plt.show()
    plt.savefig(f"result(sweeped)_CDF_[{n_ap}].{n_user}.{n_pilot}.seed{n_seed}.pdf")
    plt.savefig(f"result(sweeped)_CDF_[{n_ap}].{n_user}.{n_pilot}.seed{n_seed}.png")
    return 

if __name__=="__main__":
    main()
    plt.show()