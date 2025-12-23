from datasets.ontouch_dataset import DepthToTouchDataset


# def augment_set_with_digit_touches(input_set_path: str, output_set_path: str):
if __name__ == "__main__":

    dataset = DepthToTouchDataset(
        base_directory_address="/mnt/dataset_drive/ayad/data/recordings/soft_strawberries/",
        contact_max_m=0.04,
        far_clip_m=10.0,
        side="both",
        require_both_contact=True,
        subtract_background=True,
        rotate_180=True,
        use_colormap=False,
        resize_to=None,
        input_transform=None,
        target_transform=None,
    )
    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        breakpoint()
        depth_patch, contact_patch, meta = dataset[i]
        print(meta)
        print(f"Sample {i}: depth patch shape: {depth_patch.shape}, contact patch shape: {contact_patch.shape}")