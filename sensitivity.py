from csv import reader
import numpy as np
import matplotlib.pyplot as plt

methods = ['ngd', 'kfac', 'ekfac', 'kbfgsl']
# method = 'sgd'

damping = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
# damping = ['none']

lr = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
wd = [0.001, 0.003, 0.01, 0.03]

num_setting = 3
xy = [[wd, lr, damping], [damping, lr, wd], [damping, wd, lr]]

x_y_ = [['wd', 'lr', 'damping'], ['damping', 'lr', 'wd'], ['damping', 'wd', 'lr']]

for method in methods:
	train_acc, test_acc = [[] for d in damping], [[] for d in damping]

	# train_acc is a 3d array: [damping, lr, wd]
	for d in range(len(damping)):
		with open(method + '_params/' + method + ' - ' + str(damping[d]) + '.csv', 'r') as f:
			r = reader(f)
			for l in r:
				if l[0] == 'train acc':
					train_acc[d].append([float(c.split("/")[0]) if (c and '-' not in c) else 0.0 for c in l[1:]])
				elif l[0] == 'test acc':
					test_acc[d].append([float(c) if (c and '-' not in c) else 0.0 for c in l[1:]])
	print("===" + method + "===")

	for i in range(num_setting):
		x, y, fixed = xy[i]
		x_, y_, fixed_ = x_y_[i]

		print('- x axis = ', x_)
		print('- y axis = ', y_)
		print('- fixed = ', fixed_)


		for acc in [train_acc, test_acc]:
			if i == 0:
				z = np.array(acc) # [damping, lr, wd]
			elif i == 1:
				z = np.transpose(np.array(acc), (2,1,0)) # [wd, lr, damping]
			elif i == 2:
				z = np.transpose(np.array(acc), (1,2,0)) # [lr, wd, damping]
			else:
				raise ValueError

			for d in range(len(fixed)):
				print(fixed_ + " = " + str(fixed[d]))
				fig, ax = plt.subplots()
				ax.set_title(fixed_ + ' = ' + str(fixed[d]))

				plt.xlabel(x_)
				plt.ylabel(y_)

				# uneven spaced tick
				xspace = np.linspace(0, len(x) - 1, len(x), endpoint=True)
				yspace = np.linspace(0, len(y) - 1, len(y), endpoint=True)
				plt.xticks(xspace, [str(v) for v in x])
				plt.yticks(yspace, [str(v) for v in y])

				c = ax.pcolormesh(xspace, yspace, z[d], cmap='coolwarm', vmin=np.min(z[d]), vmax=np.max(z[d]), shading='gouraud')
				ax.axis([np.min(xspace), np.max(xspace), np.min(yspace), np.max(yspace)])
				fig.colorbar(c, ax=ax)
				plt.show()

				fig.savefig(method + '_params/' + method + '_' + fixed_ + '=' + str(fixed[d]) + '_x=' + x_ + '_y=' + y_ + '.png')

