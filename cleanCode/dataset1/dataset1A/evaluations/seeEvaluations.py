import numpy as np
import matplotlib.pyplot as plt


pearson_adam_tanh = np.load('adam_tanh/pearson.npy', allow_pickle=True)
r2_adam_tanh = np.load('adam_tanh/r2.npy', allow_pickle=True)
mse_adam_tanh = np.load('adam_tanh/mse.npy', allow_pickle=True)
visual_score_adam_tanh = np.load('adam_relu/visual_score.npy', allow_pickle=True)
visual_score_adam_tanh = [int(i) for i in visual_score_adam_tanh]

pearson_adam_relu = np.load('adam_relu/pearson.npy', allow_pickle=True)
r2_adam_relu = np.load('adam_relu/r2.npy', allow_pickle=True)
mse_adam_relu = np.load('adam_relu/mse.npy', allow_pickle=True)
visual_score_adam_relu = np.load('adam_logistic/visual_score.npy', allow_pickle=True)
visual_score_adam_relu = [int(i) for i in visual_score_adam_relu]

pearson_adam_logistic = np.load('adam_logistic/pearson.npy', allow_pickle=True)
r2_adam_logistic = np.load('adam_logistic/r2.npy', allow_pickle=True)
mse_adam_logistic = np.load('adam_logistic/mse.npy', allow_pickle=True)
visual_score_adam_logistic = np.load('adam_logistic/visual_score.npy', allow_pickle=True)
visual_score_adam_logistic = [int(i) for i in visual_score_adam_logistic]

pearson_lbfgs_logistic = np.load('lbfgs_logistic/pearson.npy', allow_pickle=True)
r2_lbfgs_logistic = np.load('lbfgs_logistic/r2.npy', allow_pickle=True)
mse_lbfgs_logistic = np.load('lbfgs_logistic/mse.npy', allow_pickle=True)
visual_score_lbfgs_logistic = np.load('lbfgs_logistic/visual_score.npy', allow_pickle=True)
visual_score_lbfgs_logistic = [int(i) for i in visual_score_lbfgs_logistic]

pearson_lbfgs_tanh = np.load('lbfgs_tanh/pearson.npy', allow_pickle=True)
r2_lbfgs_tanh = np.load('lbfgs_tanh/r2.npy', allow_pickle=True)
mse_lbfgs_tanh = np.load('lbfgs_tanh/mse.npy', allow_pickle=True)
visual_score_lbfgs_tanh = np.load('lbfgs_tanh/visual_score.npy', allow_pickle=True)
visual_score_lbfgs_tanh = [int(i) for i in visual_score_lbfgs_tanh]

pearson_lbfgs_relu = np.load('lbfgs_relu/pearson.npy', allow_pickle=True)
r2_lbfgs_relu = np.load('lbfgs_relu/r2.npy', allow_pickle=True)
mse_lbfgs_relu = np.load('lbfgs_relu/mse.npy', allow_pickle=True)
visual_score_lbfgs_relu = np.load('lbfgs_relu/visual_score.npy', allow_pickle=True)
visual_score_lbfgs_relu = [int(i) for i in visual_score_lbfgs_relu]

pearson_sgd_logistic = np.load('sgd_logistic/pearson.npy', allow_pickle=True)
r2_sgd_logistic = np.load('sgd_logistic/r2.npy', allow_pickle=True)
mse_sgd_logistic = np.load('sgd_logistic/mse.npy', allow_pickle=True)
visual_score_sgd_logistic = np.load('sgd_logistic/visual_score.npy', allow_pickle=True)
visual_score_sgd_logistic = [int(i) for i in visual_score_sgd_logistic]

pearson_sgd_tanh = np.load('sgd_tanh/pearson.npy', allow_pickle=True)
r2_sgd_tanh = np.load('sgd_tanh/r2.npy', allow_pickle=True)
mse_sgd_tanh = np.load('sgd_tanh/mse.npy', allow_pickle=True)
visual_score_sgd_tanh = np.load('sgd_tanh/visual_score.npy', allow_pickle=True)
visual_score_sgd_tanh = [int(i) for i in visual_score_sgd_tanh]

print(len(pearson_adam_tanh))
print(len(pearson_adam_relu))
print(len(pearson_adam_logistic))
print(len(pearson_lbfgs_logistic))
print(len(pearson_lbfgs_tanh))
print(len(pearson_lbfgs_relu))
print(len(pearson_sgd_logistic))
print(len(pearson_sgd_tanh))

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.subplot(2, 2, 1)
plt.plot(x, pearson_adam_tanh, label='adam tanh')
plt.plot(x, pearson_adam_relu, label='adam relu')
plt.plot(x, pearson_adam_logistic, label='adam logistic')
plt.plot(x, pearson_lbfgs_logistic, label='lbfgs logistic')
plt.plot(x, pearson_lbfgs_tanh, label='lbfgs tanh')
plt.plot(x, pearson_lbfgs_relu, label='lbfgs relu')
plt.plot(x, pearson_sgd_logistic, label='sgd logistic')
plt.plot(x, pearson_sgd_tanh, label='sgd tanh')
plt.ylabel("PCC")
plt.xlabel("Test number")

plt.subplot(2, 2, 2)
plt.plot(x, r2_adam_tanh)
plt.plot(x, r2_adam_relu)
plt.plot(x, r2_adam_logistic)
plt.plot(x, r2_lbfgs_logistic)
plt.plot(x, r2_lbfgs_tanh)
plt.plot(x, r2_lbfgs_relu)
plt.plot(x, r2_sgd_logistic)
plt.plot(x, r2_sgd_tanh)
plt.ylabel("R2")
plt.xlabel("Test number")

plt.subplot(2, 2, 3)
plt.plot(x, mse_adam_tanh)
plt.plot(x, mse_adam_relu)
plt.plot(x, mse_adam_logistic)
plt.plot(x, mse_lbfgs_logistic)
plt.plot(x, mse_lbfgs_tanh)
plt.plot(x, mse_lbfgs_relu)
plt.plot(x, mse_sgd_logistic)
plt.plot(x, mse_sgd_tanh)
plt.ylabel("MSE")
plt.xlabel("Test number")

plt.subplot(2, 2, 4)
plt.plot(x, visual_score_adam_tanh)
plt.plot(x, np.multiply(visual_score_adam_relu, 1.02))
plt.plot(x, visual_score_adam_logistic)
plt.plot(x, visual_score_lbfgs_logistic)
plt.plot(x, visual_score_lbfgs_tanh)
plt.plot(x, visual_score_lbfgs_relu)
plt.plot(x, visual_score_sgd_logistic)
plt.plot(x, visual_score_sgd_tanh)
plt.ylabel("Visual score (0-10)")
plt.xlabel("Test number")

plt.figlegend()
plt.suptitle("Test of performance: Dataset 1A")
plt.show()


print("Adam tanh")
print("Pearson: " + str(np.average(pearson_adam_tanh)) + " R2: " + str(np.average(r2_adam_tanh)) + " MSE: " +
      str(np.average(mse_adam_tanh)) + " Visual score: " + str(np.average(visual_score_adam_tanh)))

print("Adam relu")
print("Pearson: " + str(np.average(pearson_adam_relu)) + " R2: " + str(np.average(r2_adam_relu)) + " MSE: " +
      str(np.average(mse_adam_relu)) + " Visual score: " + str(np.average(visual_score_adam_relu)))

print("Adam logistic")
print("Pearson: " + str(np.average(pearson_adam_logistic)) + " R2: " + str(np.average(r2_adam_logistic)) + " MSE: " +
      str(np.average(mse_adam_logistic)) + " Visual score: " + str(np.average(visual_score_adam_logistic)))

print("LBFGS tanh")
print("Pearson: " + str(np.average(pearson_lbfgs_tanh)) + " R2: " + str(np.average(r2_lbfgs_tanh)) + " MSE: " +
      str(np.average(mse_lbfgs_tanh)) + " Visual score: " + str(np.average(visual_score_lbfgs_tanh)))

print("LBFGS logistic")
print("Pearson: " + str(np.average(pearson_lbfgs_logistic)) + " R2: " + str(np.average(r2_lbfgs_logistic)) + " MSE: " +
      str(np.average(mse_lbfgs_logistic)) + " Visual score: " + str(np.average(visual_score_lbfgs_logistic)))

print("LBFGS relu")
print("Pearson: " + str(np.average(pearson_lbfgs_relu)) + " R2: " + str(np.average(r2_lbfgs_relu)) + " MSE: " +
      str(np.average(mse_lbfgs_relu)) + " Visual score: " + str(np.average(visual_score_lbfgs_relu)))

print("SGD logistic")
print("Pearson: " + str(np.average(pearson_sgd_logistic)) + " R2: " + str(np.average(r2_sgd_logistic)) + " MSE: " +
      str(np.average(mse_sgd_logistic)) + " Visual score: " + str(np.average(visual_score_sgd_logistic)))

print("SGD tanh")
print("Pearson: " + str(np.average(pearson_sgd_tanh)) + " R2: " + str(np.average(r2_sgd_tanh)) + " MSE: " +
      str(np.average(mse_sgd_tanh)) + " Visual score: " + str(np.average(visual_score_sgd_tanh)))

pearson_high = pearson_lbfgs_relu = np.load('lbfgs_relu/pearson_good.npy', allow_pickle=True)
pearson_high_medium = pearson_lbfgs_relu = np.load('lbfgs_relu/pearson.npy', allow_pickle=True)
pearson_high_medium_low = pearson_lbfgs_relu = np.load('lbfgs_relu/pearson_all.npy', allow_pickle=True)

print("Dataportion results:")
print("High quality data: " + str(np.average(pearson_high)) + " High and medium quality data: " +
      str(np.average(pearson_high_medium)) + " High, medium and low quality data" +
      str(np.average(pearson_high_medium_low)))

