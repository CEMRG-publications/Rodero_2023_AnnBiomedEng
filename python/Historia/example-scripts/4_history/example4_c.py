from Historia.history import hm
from Historia.shared.design_utils import read_labels
from Historia.shared.plot_utils import get_col, interp_col, plot_pairwise_waves


def main():
	path_to_waves = "data/waves/"
	waves_list = [2, 3, 5, 7]  # choose the set of waves to plot

	XL = []
	for counter, idx in enumerate(waves_list):
		print(f"\nLoading wave {idx}...")
		W = hm.Wave()
		W.load(path_to_waves + f"wave_{idx}")
		W.print_stats()

		# if counter == 0:  # if wave_1 is provided, you can uncomment the following to also plot the initial space (normally a huge LHD)
		# 	X_test = W.reconstruct_tests()
		# 	XL.append(X_test)

		XL.append(W.NIMP)  # we plot the NIMP of each of the selected waves

	# colors = interp_col(get_col("blue"), len(waves_list) + 1)
	colors = interp_col(get_col("blue"), len(waves_list))

	xlabels = [f"p{i+1}" for i in range(9)]
	
	# wnames = ["Initial space"] + [f"wave {idx}" for i in waves_list]
	wnames = [f"wave {idx}" for i in waves_list]
	
	plot_pairwise_waves(XL, colors, xlabels, wnames)


if __name__ == '__main__':
	main()