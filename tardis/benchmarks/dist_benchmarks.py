#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

def main(data_set: str,
         model_checkpoint: str):
    """
    Standard benchmark for DIST on medical and standard point clouds

    ToDo: Identified standard location for test data and sanity_checks
    ToDo: Retrieve json with best CNN metric from S3 bucket
    ToDo: Build CNN from checkpoint or accept model

    ToDo: Run benchmark on standard data
    ToDo: For each data calculate F1, AP-25, AP-50, AP-75
    ToDo: Get mean value for each metric

    ToDo: Check if json have metric for tested dataset
    ToDo: Check if metrics are higher. If yes update json

    ToDo: If metric higher, sent json and save .pth with model structure at standard dir

    Args:
        data_set (str): Dataset name
        model_checkpoint (str): Directory for dict with model checkpoint with
         model structure.
    """
    pass


if __name__ == '__main__':
    main()
