import matplotlib.pyplot as plt

from src.visualizations.save_function import save_func


def plot_loss_xgb(eval_result, eval_metric, save_visual=False, timestamp=None, folder_name=None):
    for metric in eval_metric:
        epochs = len(eval_result['eval'][metric])
        x_axis = range(0, epochs)
        # plot loss
        fig, ax = plt.subplots()
        ax.plot(x_axis, eval_result['train'][metric], label='Train')
        ax.plot(x_axis, eval_result['eval'][metric], label='Test')
        ax.legend()
        plt.ylabel(metric)
        plt.title(f'XGBoost {metric}')
        save_func(save_visual=save_visual, timestamp=timestamp, folder_name=folder_name, filename='XGB_loss')
        plt.show()


def plot_loss_dnn(history, model_name, loss, metrics, save_visual=False, timestamp=None, folder_name=None):
    plt.rcParams.update({'figure.figsize': (13, 7), 'figure.dpi': 240})  # Set figure details
    total_length = 1 + len(metrics)
    fig, axs = plt.subplots(total_length, 1, constrained_layout=True)
    fig.suptitle(model_name, fontsize=16)

    axs[0].plot(history.history[loss])
    axs[0].plot(history.history[f"val_{loss}"])
    axs[0].set_title("model loss")
    axs[0].set_ylabel('loss', fontsize="large")
    axs[0].set_xlabel("epoch", fontsize="large")
    axs[0].legend(["train", "val"], loc="best")
    for i in range(1, total_length):
        metric = metrics[i - 1]
        axs[i].plot(history.history[metric])
        axs[i].plot(history.history[f"val_{metric}"])
        axs[i].set_title(f"model {metric}")
        axs[i].set_ylabel('accuracy', fontsize="large")
        axs[i].set_xlabel("epoch", fontsize="large")
        axs[i].legend(["train", "val"], loc="best")
    save_func(save_visual=save_visual, timestamp=timestamp, folder_name=folder_name, filename='model_loss')
    plt.show()
    plt.close()
