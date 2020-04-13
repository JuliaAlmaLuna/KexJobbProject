import dataManipulationFunctions as dmf


file_folder = "Julia/ecg_folder/Pat"
ecg_suffix = "_3Trace.txt"
avi_suffix1 = "_Vi.avi"

videoList = ["Julia/ecg_folder/Pat3_Vi.avi", "Julia/ecg_folder/Pat4_Vi.avi"]
ecgList = ["Julia/ecg_folder/Pat3_3Trace.txt", "Julia/ecg_folder/Pat4_3Trace.txt"]

#n_video_list = dmf.listOfVidsToListOfNestedPixelList(videoList)

n_video_list, ecg_list = dmf.createVidInputsAndTargetEcgs(videoList, ecgList)


print(len(n_video_list))
print(len(n_video_list[0]))
print(type(n_video_list))
print(type(n_video_list[0]))
a = n_video_list
b = n_video_list[0]
c = n_video_list[0][0]
d = n_video_list[0][0][0]
print(type(a))
print(type(b))
print(type(c))
print(type(d))
print(type(n_video_list[0][0][0]))

print(len(ecg_list))
print(len(ecg_list[0]))
'''
mlp = MLPRegressor(
    hidden_layer_sizes=(40,), activation='tanh', solver='lbfgs', alpha=0.12, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.2, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=99, tol=0.00001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.1, beta_1=0.2, beta_2=0.3, epsilon=1e-08)

mlp.fit(training_inputs, training_targets)
graph_predictions(mlp, testing_inputs=testing_inputs, testing_targets=testing_targets, x=X, rows=5, columns=7)
evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets)

'''

'''
imgs = dmf.vidToNestedPixelList("Pat7_Vi7.avi")
derivImgs = dmf.imgToDerivateOfImg(imgs)


print(len(imgs))
print(len(imgs[0]))
print(len(imgs[1]))
print(len(derivImgs))
print(len(derivImgs[0]))
print(len(derivImgs[1]))
print(derivImgs[1][2])
print(derivImgs[1][3])
print(derivImgs[1][200])
print(derivImgs[1][2000])
print(derivImgs[1][3000])
'''



