def startLbfgs(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    mlp.fit(training_inputs_, training_targets_)

    # This one changes so much, and does it without knowing how well it will fare with the different parameters. Should we really put it first, or in this function. Maybe a separate function
    #activation = find_best_activation(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    #mlp.set_params(activation=activation)
    # I think we have to try to optimize for each activation and see which one scores best!


   # alpha = find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
   # mlp.set_params(alpha=alpha)

  #  tolerance = find_best_tolerance(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
  #  mlp.set_params(tol=tolerance) #Needs to go through different values from now

    #layer_size = find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
   # mlp.set_params(hidden_layer_sizes=layer_size)

    epoch = find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(max_iter=epoch)




    return mlp


def graph_predictions_multi_x(mlp, testing_inputs, testing_targets, x, rows, columns, videoNames):
    size = len(testing_inputs)
    print(len(testing_inputs))
    if rows*columns < size:
        return "Graph rows/columns too few"
    else:
        for index in range(size):
            print("ohgo")
            prediction = mlp.predict(testing_inputs[index, :].reshape(1, -1))
            true = testing_targets[index, :]
            ax1 = plt.subplot(rows, columns, index+1)

            #ax1.plot(x[index, :], testing_inputs[index], 'y')
            ax1.set_title(videoNames[index])
            #ax2 = ax1.twinx()
            ax1.plot(x[index, :], true, 'g')
            ax1.plot(x[index, :], prediction[0, :], 'r')


        plt.show()
        print("hello")