#!/bin/bash
## Objective
## Assemble predictors for various data combinations and train U-Net models

# Packages
import os
import sys

#######################################################
#######################################################
# Clip path to all helper functions
function_path = os.path.expanduser("~/geoscience/albedo_downscaling/functions")
sys.path.append(function_path)
# import all the helper functions.
from albedo_unet1_fxns import *

# keep seeds fixed
os.environ["PYTHONHASHSEED"]="0"
random.seed(0); np.random.seed(0); tf.random.set_seed(0)
#######################################################
#######################################################
def main():
    
    """
    Main pipeline to preprocess satellite albedo data (GOES and MODIS), prepare training/validation/test sets,
    and train a U-Net model for blue-sky albedo prediction.

    Steps:
    1. Creates necessary directories.
    2. Loads and filters invalid dates.
    3. Loads and preprocesses training, validation, and test data.
    4. Verifies data integrity (no NaNs or infs).
    5. Trains U-Net or loads a pre-trained model.
    6. Evaluates and saves results.

    Returns:
        None
    """
    
    # Create directory to store predictions
    dest_folder = "/bsuhome/tnde/scratch/felix/UNet/Unet_test_preds_modis_new/"
    os.makedirs(dest_folder, exist_ok=True)

    # Track total runtime
    start = time.time()

    # Load invalid dates and append hard-coded invalid GOES dates
    invalid_dates = convert_dates(INVALID_DATES_PATH)
    for date in INVALID_GOES_DATES_BOTH:
        if date not in invalid_dates:
            invalid_dates.append(date)

    # Define training date range
    start_date_training_data = datetime(2021, 9, 1)
    end_date_training_data   = datetime(2022, 9, 1)

    goes_masked = True  # keep as you had

    # --- training ---
    goes_train_dict, _, goes_train_paths= get_data_and_mask(start_date_training_data, end_date_training_data, 
                                                            goes_date_gate={}, invalid_dates=invalid_dates, 
                                                            is_goes=True, use_masked_goes_dir=goes_masked)

    modis_train_dict, modis_train_mask, modis_train_paths = get_data_and_mask(start_date_training_data, end_date_training_data, 
                                                                              goes_date_gate=goes_train_dict, invalid_dates=invalid_dates, 
                                                                              is_goes=False, use_masked_goes_dir=goes_masked)

    goes_training_data_4d = stack_array_4d(goes_train_dict)           # (N,H,W,1)
    modis_training_data_4d = stack_array_4d(modis_train_dict)         # (N,H,W,1)
    modis_training_mask_3d = stack_masks_3d(modis_train_mask)         # (N,H,W)

    # --- validation ---
    start_date_validation_data = datetime(2022, 9, 2)
    end_date_validation_data   = datetime(2022, 12, 31)

    goes_val_dict, _, goes_val_paths = get_data_and_mask(start_date_validation_data, end_date_validation_data, 
                                                         goes_date_gate={}, invalid_dates=invalid_dates, 
                                                         is_goes=True, use_masked_goes_dir=goes_masked)
    modis_val_dict, modis_val_mask, modis_val_paths = get_data_and_mask(start_date_validation_data, end_date_validation_data, 
                                                                        goes_date_gate=goes_val_dict, invalid_dates=invalid_dates, 
                                                                        is_goes=False, use_masked_goes_dir=goes_masked)

    goes_validation_data_4d = stack_array_4d(goes_val_dict)
    modis_validation_data_4d = stack_array_4d(modis_val_dict)
    val_mask_3d = stack_masks_3d(modis_val_mask)
    combined_validation_data = (goes_validation_data_4d, modis_validation_data_4d)

    # --- testing (non-overlapping) ---
    start_date_test_data = datetime(2023, 1, 1)
    end_date_test_data   = datetime(2023, 6, 15)

    goes_test_dict, _, goes_test_paths = get_data_and_mask(start_date_test_data, end_date_test_data, 
                                                           goes_date_gate={}, invalid_dates=invalid_dates, 
                                                           is_goes=True, use_masked_goes_dir=goes_masked)
    modis_test_dict, modis_test_mask, modis_test_paths = get_data_and_mask(start_date_test_data, end_date_test_data, 
                                                                           goes_date_gate=goes_test_dict, invalid_dates=invalid_dates, 
                                                                           is_goes=False, use_masked_goes_dir=goes_masked)
    goes_test_data_final  = stack_array_4d(goes_test_dict)
    modis_test_data_final = stack_array_4d(modis_test_dict)
    test_mask_3d          = stack_masks_3d(modis_test_mask)
    
    # Build path list in the SAME sorted order as the stacking
    modis_test_paths_sorted = [modis_test_paths[d] for d in sorted(modis_test_dict)]  # NEW
    
    with open(UNET_RESULTS + "test_dates.json", "w") as f:
        json.dump([d.strftime("%Y-%m-%d") for d in sorted(goes_test_dict)], f)

    # Sanity checks (no NaNs in inputs; targets may have 0s where masked)
    for dataset in [goes_training_data_4d, goes_validation_data_4d, goes_test_data_final]:
        assert np.isfinite(dataset).all(), "Inputs contain NaN/Inf"

    unet_dimensions = (24, 24, 1)
    os.makedirs(UNET_RESULTS, exist_ok=True)
    os.makedirs(TENSORFLOW_TRAINING_DIR, exist_ok=True)

    run_unet(
        unet_dimensions,
        load_weights_bool=False,
        load_model=False,
        goes_training_data_4d=goes_training_data_4d,
        modis_training_data_4d=modis_training_data_4d,
        modis_training_mask_3d=modis_training_mask_3d,
        combined_validation_data=combined_validation_data,
        val_mask_3d=val_mask_3d,
        goes_test_data_final=goes_test_data_final,
        modis_test_data_final=modis_test_data_final,
        test_mask_3d=test_mask_3d,
        start_date_training_data_str=start_date_training_data.strftime("%m-%d-%Y"),
        end_date_validation_data_str=end_date_validation_data.strftime("%m-%d-%Y"),
        start_date_test_data_str=start_date_test_data.strftime("%m-%d-%Y"),
        end_date_test_data_str=end_date_test_data.strftime("%m-%d-%Y"),
        run_mask=True,
        test_ref_paths_sorted=modis_test_paths_sorted,   # NEW
        dest_folder=dest_folder                          # NEW
    )
    print(f"Time for U-Net to run: {time.time() - start:.2f} seconds")

#######################################################
#######################################################
if __name__ == "__main__":
    main()
