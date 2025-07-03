from data_preprocessing import generate_labels, build_graph, build_stgcn_dataset, STGCNDatasetSplitByParticipant, build_rnn_dataset, load_rnn_dataset
from models import STGCNModel, RNNModel
from train_eval import train_stgcn, evaluate_stgcn, train_rnn, evaluate_rnn
from utils import setup_logger, plot_metrics, plot_confusion

import torch





params = {"num_samples":15,  # Number of participants to use for training/testing
          "test_fraction":0.2, # Fraction of participants to use for testing
          "plot_graph":False, # Whether to plot the graph structure
          "sampling_interval":0.02, # Sampling interval in seconds
          "top_k":3, # Number of nearest neighbors to consider for each node
          "dist_threshold":8.0,
          "LOOKAHEAD_SECONDS" : 10.0,  # How many seconds into the future to look for the label
          "MIN_FIXATION_DURATION" : 0.1,  # Minimum fixation duration in seconds to consider a valid fixation
          "window_size":50,
          "num_epochs":30,
          "batch_size":32,
          "learning_rate":0.001,
          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),




          "model_name":"rnn",  # "stgcn", "rnn"
          
          }



def main():
    ## ------------- STGCN -----------------
    # setup_logger(**params)
    # # Step1: Generate labels and build graph
    # generate_labels(**params)
    # G, object_names, df_critical = build_graph(**params)
    # build_stgcn_dataset(df_critical, G, **params)

    # # Step2: Create dataset and split by participant, Train and Test
    # split_loader = STGCNDatasetSplitByParticipant("processed_data/stgcn_dataset_{}_{}.pt".format(params["dist_threshold"], params["LOOKAHEAD_SECONDS"]))
    # train_dataset, test_dataset, edge_index = split_loader.get_datasets()
    # object_names = split_loader.object_names
    # model = STGCNModel(num_nodes=len(object_names))
    # history = train_stgcn(model, train_dataset, test_dataset, edge_index, **params)
    # all_labels, all_preds = evaluate_stgcn(model, test_dataset, edge_index, **params)
    # plot_metrics(history, **params)
    # plot_confusion(all_labels, all_preds, **params)


    ## ------------- RNN -----------------
    setup_logger(**params)
    # Step1: Generate and preprocess RNN dataset
    generate_labels(**params)
    build_rnn_dataset(output_path="processed_data", **params)

    # Step2: Create dataset and split by participant, Train and Test
    train_dataset, test_dataset, pos_weight, input_dim = \
    load_rnn_dataset("processed_data/rnn_dataset_{}_{}_{}_{}_participants.pt".format(params["dist_threshold"], params["LOOKAHEAD_SECONDS"], params["MIN_FIXATION_DURATION"], int(params["num_samples"])), **params)
    model = RNNModel(in_dim=input_dim).to(params["device"])
    history = train_rnn(model, train_dataset, test_dataset, pos_weight, **params)
    all_labels, all_preds = evaluate_rnn(model, test_dataset, **params)
    plot_metrics(history, **params)
    plot_confusion(all_labels, all_preds, **params)




if __name__ == "__main__":
    main()
