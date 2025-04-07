import os

def retrieve_data_sample_path(
    data_source,
    experiment_name,
    experiment_specifics,
    data_ontology,
    sample_id,
    sub_data_sample_id=None,
    data_sample_name=None,
    data_extension=None,
):
    """
    Function to retrieve the data sample from the specified path.
    """
    # Construct the path to the data sample
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_sample_name = (
        f"{data_sample_name}_{sample_id}_{sub_data_sample_id}"
        if sub_data_sample_id
        else f"{data_sample_name}_{sample_id}"
    )
    data_extension = "obj"

    path = os.path.join(
        repository_root,
        "assets",
        "data",
        data_source,
        experiment_name,
        experiment_specifics,
        data_ontology,
        f"{data_sample_name}.{data_extension}",
    )

    return path