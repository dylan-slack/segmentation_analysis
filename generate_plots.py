import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['figure.figsize'] = 15,8
rcParams['font.size'] = 22.0

q_df = pd.read_csv("quickshift.csv", index_col=0)
f_df = pd.read_csv("felzenszwalb.csv", index_col=0)
s_df = pd.read_csv("slic.csv", index_col=0)


def gen_seg_specific(df, name):
	# df = df[df['img_num'] <= 50]

	ax = sns.boxplot(x="img_num", y="fidelity", data=df).set(xlabel='image', title=f"Superpixel function: {name}", xticks=[])
	# plt.ylim(0, 1)
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.savefig(name + '_imagenum_fidelity_boxplot.png')
	plt.cla()

	ax = sns.regplot(x="n_segs", y="fidelity", data=df).set(title=name, xlabel='number_of_superpixels')
	plt.ylim(0, 1)

	plt.savefig(name + 'n_segs_fidelity_scatterplot.png')
	plt.cla()

	# ax = sns.lineplot(x="n_segs", y="fidelity", hue="img_num", data=df).set(title=name, xlabel='number_of_superpixels')
	# plt.ylim(0, 1)

	# plt.savefig(name + 'n_segs_fidelity_lineplot.png')
	# plt.cla()


	variances = []
	for img_num in range(0, 50):
		variances.append(np.std(df[df["img_num"] == img_num]["fidelity"]))

	variances = np.array(variances)
	print(name)
	print(variances[np.argsort(variances)[-10:]])
	print(np.argsort(variances)[-10:])


gen_seg_specific(q_df, "quickshift")
gen_seg_specific(f_df, "felzenszwalb")
gen_seg_specific(s_df, "slic")